"""Runtime-aligned selector loss and metrics for DFlash2.

``eal`` follows the realized greedy selector path. ``accept_len`` remains the
analytical TV-overlap estimate over the unary logits.
"""

from collections.abc import Callable
from typing import Any

import torch
from torch.nn import functional

from speculators.losses import (
    LossConfig,
    dflash_loss_decay,
    dpace_loss_decay,
    tv_loss,
)
from speculators.losses.utils import _LOSS_REDUCTION_EPS
from speculators.models.dspark.metrics import compute_metrics as compute_unary_metrics
from speculators.models.metrics import compute_accepted_length_counts

__all__ = [
    "compute_metrics",
    "compute_selector_loss",
    "selector_training_candidates",
    "strict_topk_selector_targets",
]


def selector_training_candidates(
    candidate_ids: torch.Tensor,  # [*, top_k]
    target_ids: torch.Tensor,  # [*]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build top-k selector candidates, injecting a missing target at rank K.

    Unary top-k is the serving candidate set. Training replaces its weakest
    candidate only when the hard target is absent, so every position has a
    well-defined K-way cross-entropy label without expanding the selector to the
    full vocabulary.

    Returns (training_candidate_ids, target_positions, contains_target) where
    contains_target is a boolean mask indicating positions where the target was
    already present in the original unary top-k.
    """
    top_k = candidate_ids.shape[-1]
    target_matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    contains_target = target_matches.any(dim=-1)
    target_positions = target_matches.to(torch.int64).argmax(dim=-1)
    target_positions = torch.where(
        contains_target,
        target_positions,
        top_k - 1,
    )

    training_candidate_ids = candidate_ids.clone()
    training_candidate_ids[..., -1] = torch.where(
        contains_target,
        training_candidate_ids[..., -1],
        target_ids,
    )
    return training_candidate_ids, target_positions, contains_target


def strict_topk_selector_targets(
    candidate_ids: torch.Tensor,  # [*, top_k]
    target_ids: torch.Tensor,  # [*]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Labels over the strict serving top-k, with no target injection.

    Serving re-ranks exactly the unary top-k, so the selector objective is a
    K-way classification over that set and nothing else: positions whose hard
    target fell outside the top-k are backbone/recall failures and must not
    produce selector gradient. Matches SpecForge's ``_selector_chunk_terms``
    and the published DFlash2 reproduction recipe, which found the strict
    top-k selector objective gave the best serving acceptance length.

    Returns (target_positions, contains_target); callers gate the selector
    loss with ``contains_target``.
    """
    target_matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    contains_target = target_matches.any(dim=-1)
    target_positions = target_matches.to(torch.int64).argmax(dim=-1)
    return target_positions, contains_target


def _candidate_cross_entropy(
    logits: torch.Tensor,  # [*, top_k]
    target_positions: torch.Tensor,  # [*]
) -> torch.Tensor:
    return functional.cross_entropy(
        logits.flatten(0, -2),
        target_positions.flatten(),
        reduction="none",
    ).view_as(target_positions)


def compute_selector_loss(
    candidate_logits: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    target_positions: torch.Tensor,  # [1, num_anchors*block_size]
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    block_size: int,
    *,
    gamma: float,
    per_position_loss_weight: str,
    dpace_alpha: float,
    sample_from_anchor: bool = False,
) -> torch.Tensor:
    """Compute teacher-forced hard CE over the runtime-sized candidate set.

    The numerator is decay-weighted exactly like the unary objective, and the
    denominator is the same decay-weighted mask mass the reference
    implementations (TorchSpec, SpecForge) normalize by — not the undecayed
    mask count. Normalizing by the undecayed count would silently scale the
    selector term by ``mask.sum() / (mask * decay).sum()`` (~1.5-2x for the
    usual gamma/block-size combinations) relative to ``selector_loss_alpha``.
    The CE runs in fp32 for numerical parity with the references.
    """
    pos_idx = (
        torch.arange(candidate_logits.shape[1], device=candidate_logits.device)
        % block_size
    ).unsqueeze(0)
    elementwise_loss = _candidate_cross_entropy(
        candidate_logits.float(), target_positions
    )
    loss_mask = loss_mask.to(elementwise_loss.dtype)
    elementwise_loss = elementwise_loss * loss_mask

    if per_position_loss_weight == "dpace":
        decay_mult = dpace_loss_decay(
            pos_idx.to(elementwise_loss.dtype),
            loss_mask=loss_mask,
            block_size=block_size,
            dpace_alpha=dpace_alpha,
            elementwise_loss=elementwise_loss,
        )
    else:
        decay_mult = dflash_loss_decay(
            pos_idx.to(elementwise_loss.dtype),
            gamma=gamma,
            sample_from_anchor=sample_from_anchor,
        )
    weighted_loss = elementwise_loss * decay_mult
    denominator = (loss_mask * decay_mult).sum(dim=1) + _LOSS_REDUCTION_EPS
    return (weighted_loss.sum(dim=1) / denominator).mean()


def compute_metrics(
    unary_logits: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    targets: torch.Tensor,  # [1, num_anchors*block_size, draft_vocab_size]
    training_candidate_ids: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    candidate_logits: torch.Tensor,  # [1, num_anchors*block_size, top_k]
    target_positions: torch.Tensor,  # [1, num_anchors*block_size]
    contains_target: torch.Tensor,  # [1, num_anchors*block_size]
    loss_mask: torch.Tensor,  # [1, num_anchors*block_size]
    block_size: int,
    top_k: int,
    sample_from_anchor: bool = False,
    *,
    loss_config: LossConfig,
    tv_loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = tv_loss,
    gamma: float = 4.0,
    selector_loss_alpha: float = 1.0,
    per_position_loss_weight: str = "fixed-exp-decay",
    dpace_alpha: float = 0.5,
    selector_loss_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Combine the unary DFlash objective with a K-way selector objective.

    ``selector_loss_mask`` gates which positions produce selector gradient;
    under the strict top-k objective it is ``loss_mask * contains_target`` so
    coverage misses (a backbone/recall failure) teach the selector nothing.
    Defaults to ``loss_mask``, matching the candidate-injection ablation.
    """
    if selector_loss_mask is None:
        selector_loss_mask = loss_mask
    unary_loss, metrics = compute_unary_metrics(
        unary_logits,
        targets,
        None,
        loss_mask,
        block_size,
        loss_config=loss_config,
        tv_loss_fn=tv_loss_fn,
        gamma=gamma,
        confidence_head_alpha=0.0,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )
    selector_loss = compute_selector_loss(
        candidate_logits,
        target_positions,
        selector_loss_mask,
        block_size,
        gamma=gamma,
        per_position_loss_weight=per_position_loss_weight,
        dpace_alpha=dpace_alpha,
        sample_from_anchor=sample_from_anchor,
    )
    loss = unary_loss + selector_loss_alpha * selector_loss

    one = torch.ones((), device=unary_logits.device)
    metrics["unary_loss_sum"] = unary_loss.detach().clone()
    metrics["unary_loss_total"] = one
    metrics["selector_loss_sum"] = selector_loss.detach().clone()
    metrics["selector_loss_total"] = one.clone()
    metrics["loss_sum"] = loss.detach().clone()
    metrics["loss_total"] = one.clone()

    with torch.no_grad():
        target_ids = targets.argmax(dim=-1)
        valid = loss_mask.to(torch.bool)
        valid_float = valid.to(unary_logits.dtype)
        valid_total = valid_float.sum()

        metrics[f"unary_candidate_recall_at_{top_k}_sum"] = (
            contains_target.to(valid_float.dtype) * valid_float
        ).sum()
        metrics[f"unary_candidate_recall_at_{top_k}_total"] = valid_total

        target_log_normalizer = torch.logsumexp(targets.float(), dim=-1)
        candidate_target_logits = targets.gather(-1, training_candidate_ids).float()
        candidate_mass = torch.exp(
            torch.logsumexp(candidate_target_logits, dim=-1) - target_log_normalizer
        )
        metrics[f"unary_candidate_target_mass_at_{top_k}_sum"] = (
            candidate_mass * valid_float
        ).sum()
        metrics[f"unary_candidate_target_mass_at_{top_k}_total"] = valid_total.clone()

        teacher_forced_ids = training_candidate_ids.gather(
            -1, candidate_logits.detach().argmax(dim=-1, keepdim=True)
        ).squeeze(-1)
        serving_valid = valid_float * contains_target.to(valid_float.dtype)
        serving_total = serving_valid.sum()
        metrics["teacher_forced_selector_acc_sum"] = (
            teacher_forced_ids.eq(target_ids).to(valid_float.dtype) * serving_valid
        ).sum()
        metrics["teacher_forced_selector_acc_total"] = serving_total

        num_blocks = unary_logits.shape[1] // block_size
        contains_target_blocks = contains_target.view(num_blocks, block_size)
        valid_blocks = valid.view(num_blocks, block_size)

        # Teacher-forced predecessor tokens are exact while the greedy path is
        # alive. Gate on the original unary candidate set because training may
        # inject a missing target that would not be available during serving.
        start_pos = 0 if sample_from_anchor else 1
        selector_correct = teacher_forced_ids.eq(target_ids) & contains_target
        eal_sum, eal_total = compute_accepted_length_counts(
            selector_correct.view(num_blocks, block_size)[:, start_pos:],
            valid_blocks[:, start_pos:],
        )
        metrics["eal_sum"] = eal_sum
        metrics["eal_total"] = eal_total

        oracle_alive = torch.ones(
            num_blocks, dtype=torch.bool, device=unary_logits.device
        )
        oracle_accepted_length = torch.ones(
            num_blocks, dtype=torch.float32, device=unary_logits.device
        )
        for position in range(1, block_size):
            oracle_alive = (
                oracle_alive
                & valid_blocks[:, position]
                & contains_target_blocks[:, position]
            )
            oracle_accepted_length += oracle_alive.to(oracle_accepted_length.dtype)
        block_valid = valid_blocks[:, 1:].any(dim=-1)
        block_total = block_valid.sum().to(torch.float32)
        metrics[f"unary_top_{top_k}_oracle_accepted_length_sum"] = (
            oracle_accepted_length * block_valid
        ).sum()
        metrics[f"unary_top_{top_k}_oracle_accepted_length_total"] = block_total

    return loss, metrics
