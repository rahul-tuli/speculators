import sys
from pathlib import Path

import pytest
import torch
from datasets import Dataset
from safetensors.torch import load_file, save_file

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dquant.quantize import make_real_loader


@pytest.fixture
def cache(tmp_path):
    Dataset.from_dict(
        {
            "input_ids": [[1, 2, 3, 4], [5, 6]],
            "loss_mask": [[0, 1, 1, 1], [1, 1]],
            "seq_len": [4, 2],
        }
    ).with_format("torch").save_to_disk(str(tmp_path))
    (tmp_path / "hidden_states").mkdir()
    for i, tokens in enumerate(([1, 2, 3, 4], [5, 6])):
        t = len(tokens)
        save_file(
            {
                "hidden_states": torch.arange(t * 3 * 2)
                .reshape(t, 3, 2)
                .to(torch.float8_e4m3fn),
                "hidden_states_scales": torch.full((t, 1, 1), 0.25),
                "token_ids": torch.tensor(tokens),
            },
            str(tmp_path / "hidden_states" / f"hs_{i}.safetensors"),
        )
    (tmp_path / "vllm_command.txt").write_text(
        "python -m vllm serve target-A --speculative_config "
        '\'{"draft_model_config": {"hf_config": '
        '{"eagle_aux_hidden_state_layer_ids": [1, 3, 5]}}}\'\n'
    )
    return tmp_path


def test_scales_alignment_and_lazy_reads(cache, monkeypatch):
    from hs_connectors.transfer import FP8Transfer

    calls = []
    original = FP8Transfer.get_cached

    def read(self, idx):
        calls.append(idx)
        return original(self, idx)

    monkeypatch.setattr(FP8Transfer, "get_cached", read)
    loader, count = make_real_loader(str(cache), 3, 2, 0)
    assert count == 2
    assert calls == []  # Construction must not retain decoded corpus tensors.
    batches = list(loader)
    assert calls == [0, 1]
    batch = batches[0]
    raw = load_file(str(cache / "hidden_states/hs_0.safetensors"))
    restored = raw["hidden_states"].bfloat16() * raw["hidden_states_scales"].bfloat16()
    assert batch["hidden_states"].dtype == torch.bfloat16
    assert batch["verifier_last_hidden_states"].dtype == torch.bfloat16
    torch.testing.assert_close(batch["hidden_states"][0], restored[:3, :-1].flatten(1))
    torch.testing.assert_close(
        batch["verifier_last_hidden_states"][0], restored[:3, -1]
    )
    assert batch["input_ids"].tolist() == [[1, 2, 3]]
    assert batch["loss_mask"].tolist() == [[0, 1, 1]]
    assert torch.isfinite(torch.amin(batch["hidden_states"], dim=(0, -1))).all()
    assert loader.dataset.manifest["actual_tokens"] == 5
    assert loader.dataset.manifest["requested_token_capacity"] == 6


def test_budget_shortfall_is_explicit(cache):
    with pytest.raises(ValueError, match="requested 3.*available 2"):
        make_real_loader(str(cache), 3, 3, 0)


@pytest.mark.parametrize(
    "corruption", ["missing", "scales", "tokens", "nan", "zero_scale", "length", "mask"]
)
def test_invalid_cache_fails_closed(cache, corruption):
    p = cache / "hidden_states/hs_0.safetensors"
    raw = load_file(str(p))
    if corruption == "missing":
        p.unlink()
    else:
        if corruption == "scales":
            del raw["hidden_states_scales"]
        elif corruption == "tokens":
            raw["token_ids"][0] = 99
        elif corruption == "nan":
            raw["hidden_states_scales"][0] = float("nan")
        elif corruption == "zero_scale":
            raw["hidden_states_scales"][0] = 0
        elif corruption == "length":
            raw["hidden_states"] = raw["hidden_states"][:2].clone()
        elif corruption == "mask":
            Dataset.from_dict(
                {
                    "input_ids": [[1, 2, 3, 4], [5, 6]],
                    "loss_mask": [[1], [1, 1]],
                    "seq_len": [4, 2],
                }
            ).with_format("torch").save_to_disk(str(cache))
        save_file(raw, str(p))
    with pytest.raises(
        (ValueError, RuntimeError), match="sample|cache|FP8|finite|mask"
    ):
        loader, _ = make_real_loader(str(cache), 3, 2, 0)
        list(loader)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"expected_target": "target-B"},
        {"expected_layers": [3, 1]},
        {"expected_hidden_size": 8},
    ],
)
def test_incompatible_model_rejected(cache, kwargs):
    with pytest.raises(ValueError, match="mismatch"):
        make_real_loader(str(cache), 3, 2, 0, **kwargs)


def test_unquantized_storage_and_repeatable_selection(cache):
    for file in (cache / "hidden_states").glob("*.safetensors"):
        raw = load_file(str(file))
        raw["hidden_states"] = (
            raw["hidden_states"].bfloat16() * raw.pop("hidden_states_scales").bfloat16()
        )
        save_file(raw, str(file))
    loader, _ = make_real_loader(str(cache), 4, 2, 7)
    repeated, _ = make_real_loader(str(cache), 4, 2, 7)
    assert (
        loader.dataset.manifest["selected_indices"]
        == repeated.dataset.manifest["selected_indices"]
    )
    for first, second in zip(loader, repeated, strict=True):
        assert first["hidden_states"].dtype == torch.bfloat16
        torch.testing.assert_close(first["hidden_states"], second["hidden_states"])


def test_random_data_is_lazy_repeatable_and_shaped_like_drafter():
    from types import SimpleNamespace

    from dquant.quantize import RandomBatchDataset

    model = SimpleNamespace(
        fc=SimpleNamespace(in_features=4),
        verifier_lm_head=SimpleNamespace(weight=torch.zeros(8, 2)),
        embed_tokens=SimpleNamespace(num_embeddings=8),
    )
    dataset = RandomBatchDataset(model, 3, 4, 9, torch.bfloat16)
    a, b = dataset[1], dataset[1]
    assert a["hidden_states"].shape == (1, 4, 4)
    assert a["verifier_last_hidden_states"].shape == (1, 4, 2)
    for key in a:
        torch.testing.assert_close(a[key], b[key])
    assert not torch.equal(dataset[0]["hidden_states"], a["hidden_states"])
