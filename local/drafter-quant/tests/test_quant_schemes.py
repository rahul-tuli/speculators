"""Check that the added recipes request real GPTQ and observed importance."""

# ruff: noqa: INP001, S101, SLF001, PLR2004
# These are standalone experiment tests under local/, outside repo test ignores.

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dquant.quantize import build_recipe
from llmcompressor.modifiers.gptq import GPTQModifier
from llmcompressor.observers import Observer
from llmcompressor.recipe import Recipe


def test_gptq_pair_differs_only_by_importance_observer():
    ordinary = build_recipe("nvfp4_gptq", ["lm_head"])
    imatrix = build_recipe("nvfp4_gptq_imatrix", ["lm_head"])
    assert isinstance(ordinary, GPTQModifier)
    assert isinstance(imatrix, GPTQModifier)
    assert ordinary.ignore == imatrix.ignore == ["lm_head"]
    a = ordinary.resolved_config.config_groups["group_0"]
    b = imatrix.resolved_config.config_groups["group_0"]
    assert a.input_activations == b.input_activations
    assert (
        a.weights.model_copy(
            update={
                "observer": b.weights.observer,
                "observer_kwargs": b.weights.observer_kwargs,
            }
        )
        == b.weights
    )
    assert a.weights.observer == "nvfp4_expanded_mse"
    assert b.weights.observer == "nvfp4_expanded_imatrix"
    assert b.weights.observer_kwargs == {"strict": True}


def test_imatrix_requires_observed_inputs():
    scheme = build_recipe("nvfp4_gptq_imatrix", []).resolved_config.config_groups[
        "group_0"
    ]
    layer = torch.nn.Linear(16, 16, bias=False)
    observer = Observer.load_from_registry(
        scheme.weights.observer, base_name="weight", args=scheme.weights
    )
    observer.attach(layer)
    with pytest.raises(ValueError, match="no importance data"):
        observer(layer.weight)
    layer(torch.randn(4, 16))
    observer(layer.weight)
    assert observer._imatrix_count.item() == 4
    assert torch.isfinite(observer._imatrix_sum).all()
    observer.detach(layer)


@pytest.mark.parametrize("scheme", ["nvfp4_gptq", "nvfp4_gptq_imatrix"])
def test_requested_dampening_is_serialized(scheme):
    modifier = build_recipe(scheme, [], gptq_dampening_frac=0.1)
    assert modifier.dampening_frac == 0.1
    assert "dampening_frac: 0.1" in Recipe.create_instance(modifier).yaml()
