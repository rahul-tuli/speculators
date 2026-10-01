from collections.abc import Callable
from typing import Any, ClassVar

import torch
from transformers import PretrainedConfig

from speculators.losses import LossConfig, kl_div_loss, resolve_loss_config, tv_loss
from speculators.model import SpeculatorModel
from speculators.models.dflash.config import DFlashSpeculatorConfig
from speculators.models.dflash.core import DFlashDraftModel
from speculators.models.dflash2.config import DFlash2SpeculatorConfig
from speculators.models.dflash2.metrics import (
    compute_metrics,
    selector_training_candidates,
    strict_topk_selector_targets,
)
from speculators.models.dflash2.model_definitions import (
    CandidateSelector,
    Qwen3DFlash2DecoderLayer,
)
from speculators.models.utils import conditional_torch_compile

__all__ = [
    "DFlash2DraftModel",
]

_DEFAULT_LOSS_CONFIG: LossConfig = {"kl_div": (kl_div_loss, 1.0)}

_SELECTOR_CANDIDATE_MODES = ("strict-topk", "inject")


@SpeculatorModel.register("dflash2")
class DFlash2DraftModel(DFlashDraftModel):
    """DFlash with local convolution and bilinear candidate reranking."""

    config_class: ClassVar[type[DFlash2SpeculatorConfig]] = DFlash2SpeculatorConfig  # type: ignore[misc,assignment]
    _no_split_modules = ["Qwen3DFlash2DecoderLayer"]

    def __init__(self, config: DFlash2SpeculatorConfig) -> None:
        target_vocab_size = config.transformer_layer_config.vocab_size
        if config.draft_vocab_size != target_vocab_size:
            raise ValueError(
                "DFlash2 currently requires the full verifier vocabulary: "
                f"draft_vocab_size={config.draft_vocab_size}, "
                f"verifier_vocab_size={target_vocab_size}."
            )
        if config.selector_top_k > target_vocab_size:
            raise ValueError(
                f"selector_top_k ({config.selector_top_k}) cannot exceed "
                f"verifier_vocab_size ({target_vocab_size})."
            )
        super().__init__(config=config)

        for layer_ in self.layers:
            assert isinstance(layer_, Qwen3DFlash2DecoderLayer)  # noqa: S101
            layer_.reset_convolutions()

        initializer_range = getattr(
            config.transformer_layer_config, "initializer_range", 0.02
        )
        self.candidate_selector = CandidateSelector(
            vocab_size=target_vocab_size,
            hidden_size=config.transformer_layer_config.hidden_size,
            rank=config.selector_rank,
            top_k=config.selector_top_k,
            initializer_range=initializer_range,
        )

    def _make_decoder_layer(
        self, config: DFlashSpeculatorConfig, layer_idx: int
    ) -> Qwen3DFlash2DecoderLayer:
        assert isinstance(config, DFlash2SpeculatorConfig)  # noqa: S101
        return Qwen3DFlash2DecoderLayer(
            config.transformer_layer_config,  # type: ignore[arg-type]
            layer_idx,
            block_size=config.block_size,
            conv_kernel_size=config.conv_kernel_size,
            conv_group_size=config.conv_group_size,
        )

    @classmethod
    def from_training_args(
        cls,
        verifier_config: PretrainedConfig,
        t2d: torch.Tensor | None = None,
        d2t: torch.Tensor | None = None,
        **kwargs,
    ) -> "DFlash2DraftModel":
        """Create a DFlash2 model from the shared training arguments."""
        config = DFlash2SpeculatorConfig(
            **cls._build_base_config_kwargs("dflash2", verifier_config, **kwargs),
            conv_kernel_size=kwargs.get("conv_kernel_size", 2),
            conv_group_size=kwargs.get("conv_group_size", 16),
            selector_rank=kwargs.get("selector_rank", 256),
            selector_top_k=kwargs.get("selector_top_k", 16),
            output_multiplier=kwargs.get("output_multiplier", 1.0),
            final_logit_softcapping=kwargs.get("final_logit_softcapping"),
            input_embedding_scale=kwargs.get("input_embedding_scale", 1.0),
        )
        model = cls(config=config)
        model.load_vocab_mappings(t2d, d2t)
        model.load_verifier_weights()
        return model

    @staticmethod
    def get_trainer_kwargs(**kwargs) -> tuple[dict, dict]:
        """Resolve the unary and selector objectives used during training."""
        implementation = kwargs.get("loss_implementation", "fused")
        loss_config = resolve_loss_config(kwargs["loss_fn"], implementation)
        tv_loss_fn = resolve_loss_config("tv", implementation)["tv"][0]
        shared = {
            "loss_config": loss_config,
            "tv_loss_fn": tv_loss_fn,
            # DFlash2 defaults follow the published reproduction recipe
            # (hard-target CE, loss_decay_gamma=7, strict top-k selector).
            "gamma": kwargs.get("dflash_decay_gamma", 7.0),
            "max_anchors": kwargs.get("max_anchors", 512),
            "per_position_loss_weight": kwargs.get(
                "per_position_loss_weight", "fixed-exp-decay"
            ),
            "dpace_alpha": kwargs.get("dpace_alpha", 0.5),
            "selector_loss_alpha": kwargs.get("selector_loss_alpha", 1.0),
            "selector_candidate_mode": kwargs.get(
                "selector_candidate_mode", "strict-topk"
            ),
        }
        return dict(shared), dict(shared)

    def transform_unary_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """Apply the checkpoint-contract unary transform (multiplier, softcap).

        Serving engines apply ``output_multiplier`` and
        ``final_logit_softcapping`` from ``dflash_config`` before the selector
        adds its transition scores, so training must score against the same
        transformed logits. Identity for the Qwen3.8-27B recipe (multiplier
        1.0, no softcap).
        """
        multiplier = float(self.config.output_multiplier)
        softcap = self.config.final_logit_softcapping
        if multiplier == 1.0 and softcap is None:
            return logits
        logits = logits.float() * multiplier
        if softcap is not None:
            softcap = float(softcap)
            logits = torch.tanh(logits / softcap) * softcap
        return logits

    def _scale_noise_embedding(self, noise_embedding: torch.Tensor) -> torch.Tensor:
        scale = float(self.config.input_embedding_scale)
        if scale == 1.0:
            return noise_embedding
        return noise_embedding * scale

    def _predecessor_ids(
        self,
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        anchored_block_indices: torch.Tensor,  # shape: [num_anchors*block_size]
    ) -> torch.Tensor:
        block_tokens = input_ids[0, anchored_block_indices].view(
            -1, self.block_size
        )  # shape: [num_anchors, block_size]
        if self.config.sample_from_anchor:
            return block_tokens
        # For token sequence 0 1 2 3 return previous token ids 0 0 1 2
        return torch.cat([block_tokens[:, :1], block_tokens[:, :-1]], dim=1)

    @conditional_torch_compile
    def forward(
        self,
        hidden_states: torch.Tensor,  # shape: [1, total_seq_len,num_hidden*hidden_size]
        input_ids: torch.Tensor,  # shape: [1, total_seq_len]
        loss_mask: torch.Tensor,  # shape: [1, total_seq_len]
        verifier_last_hidden_states: torch.Tensor,  # shape: [1, total_seq_len, hidden_size] # noqa: E501
        document_ids: torch.Tensor,  # shape: [1, total_seq_len]
        position_ids: torch.Tensor | None = None,  # shape: [1, total_seq_len]
        loss_config: LossConfig | None = None,
        tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
        gamma: float = 7.0,
        max_anchors: int = 512,
        selector_loss_alpha: float = 1.0,
        selector_candidate_mode: str = "strict-topk",
        per_position_loss_weight: str = "fixed-exp-decay",
        dpace_alpha: float = 0.5,
        **kwargs,
    ) -> tuple[None, torch.Tensor, dict[str, Any]]:
        hidden, unary_logits, targets, aligned_loss_mask, block_indices = (
            self._backbone_forward(
                hidden_states,
                input_ids,
                loss_mask,
                verifier_last_hidden_states,
                document_ids,
                position_ids,
                max_anchors=max_anchors,
                **kwargs,
            )
        )
        # Score against the same transformed logits serving uses (M3 contract).
        unary_logits = self.transform_unary_logits(unary_logits)
        predecessor_ids = self._predecessor_ids(input_ids, block_indices)

        target_ids = targets.argmax(dim=-1)
        # shape: [1, num_anchors*block_size]
        candidate_ids = unary_logits.topk(self.candidate_selector.top_k, dim=-1).indices
        # shape: [1, num_anchors*block_size, top_k]
        if selector_candidate_mode == "strict-topk":
            # Serving-aligned default: the selector re-ranks exactly the unary
            # top-k; coverage misses are a backbone/recall failure and produce
            # no selector gradient.
            target_positions, contains_target = strict_topk_selector_targets(
                candidate_ids, target_ids
            )
            training_candidate_ids = candidate_ids
            selector_loss_mask = aligned_loss_mask * contains_target.to(
                aligned_loss_mask.dtype
            )
        elif selector_candidate_mode == "inject":
            # Ablation: splice a missing hard target into the weakest candidate
            # slot and train the selector on every position.
            training_candidate_ids, target_positions, contains_target = (
                selector_training_candidates(candidate_ids, target_ids)
            )
            selector_loss_mask = aligned_loss_mask
        else:
            raise ValueError(
                f"Unknown selector_candidate_mode {selector_candidate_mode!r}; "
                f"expected one of {_SELECTOR_CANDIDATE_MODES}."
            )
        candidate_logits = self.candidate_selector.score_candidates(
            unary_logits,
            hidden,
            predecessor_ids.reshape(1, -1),
            training_candidate_ids,
        )

        loss, metrics = compute_metrics(
            unary_logits=unary_logits,
            targets=targets,
            training_candidate_ids=training_candidate_ids,
            candidate_logits=candidate_logits,
            target_positions=target_positions,
            contains_target=contains_target,
            loss_mask=aligned_loss_mask,
            block_size=self.block_size,
            top_k=self.candidate_selector.top_k,
            sample_from_anchor=self.config.sample_from_anchor,
            loss_config=loss_config or _DEFAULT_LOSS_CONFIG,
            tv_loss_fn=tv_loss_fn,
            gamma=gamma,
            selector_loss_alpha=selector_loss_alpha,
            per_position_loss_weight=per_position_loss_weight,
            dpace_alpha=dpace_alpha,
            selector_loss_mask=selector_loss_mask,
        )
        return None, loss, metrics
