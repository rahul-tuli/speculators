from typing import Literal

from pydantic import Field

from speculators import SpeculatorModelConfig
from speculators.models.dflash.config import DFlashSpeculatorConfig

__all__ = [
    "DFlash2SpeculatorConfig",
]


@SpeculatorModelConfig.register("dflash2")
class DFlash2SpeculatorConfig(DFlashSpeculatorConfig):
    """DFlash configuration with local convolutions and a candidate selector."""

    speculators_model_type: Literal["dflash2"] = "dflash2"  # type: ignore[assignment]
    architectures: list[str] = Field(
        default_factory=lambda: ["DFlash2DraftModel"],
        description="Model architectures that can load these weights",
    )
    sliding_window_non_causal: bool = Field(
        default=True,
        description="Use bidirectional masking inside sliding-window draft blocks.",
    )
    conv_kernel_size: int = Field(
        default=2,
        ge=1,
        description="Number of causal taps in each local dynamic convolution.",
    )
    conv_group_size: int = Field(
        default=16,
        ge=1,
        description="Number of hidden channels sharing each dynamic kernel.",
    )
    selector_rank: int = Field(
        default=256,
        ge=1,
        description="Rank of the candidate selector factorization.",
    )
    selector_top_k: int = Field(
        default=16,
        ge=1,
        description="Number of unary candidates reranked during inference.",
    )
    output_multiplier: float = Field(
        default=1.0,
        gt=0,
        description="Multiplier applied to unary logits before the loss and "
        "selector scoring. Part of the reference checkpoint contract "
        "(dflash_config.output_multiplier); e.g. Muse-Glimmer-30B-DFlash2 "
        "uses ~0.196.",
    )
    final_logit_softcapping: float | None = Field(
        default=None,
        gt=0,
        description="Tanh softcap applied to unary logits after "
        "output_multiplier (dflash_config.final_logit_softcapping). "
        "None keeps logits uncapped.",
    )
    input_embedding_scale: float = Field(
        default=1.0,
        gt=0,
        description="Scale applied to the noise embeddings fed to the draft "
        "backbone (dflash_config.input_embedding_scale in reference "
        "checkpoints). 1.0 leaves embeddings untouched.",
    )
