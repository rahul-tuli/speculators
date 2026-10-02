from __future__ import annotations

import json
import sys
from argparse import Namespace
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "evaluate"))
import point  # type: ignore[import-not-found]
from point import (  # type: ignore[import-not-found]
    acceptance,
    completed,
    digest,
    save,
    validate_requests,
)


def counters(d, p, a, positions):
    return "\n".join(
        [
            f"vllm:spec_decode_num_drafts {d}",
            f"vllm:spec_decode_num_draft_tokens {p}",
            f"vllm:spec_decode_num_accepted_tokens {a}",
            *[
                f'vllm:spec_decode_num_accepted_tokens_per_pos{{position="{i}"}} {v}'
                for i, v in enumerate(positions)
            ],
        ]
    )


def test_acceptance_excludes_warmup_and_keeps_denominators():
    result = acceptance(counters(10, 20, 10, [7, 3]), counters(14, 28, 15, [10, 5]), 2)
    assert result["acceptance_length"] == 2.25
    assert result["accepted_token_fraction"] == 5 / 8
    assert result["accepted_by_position"] == [3, 2]
    assert result["per_position_acceptance"] == [0.75, 0.5]


@pytest.mark.parametrize(
    "after",
    [
        counters(1, 2, 1, [1, 0]),
        counters(14, 28, 16, [10, 5]),
        counters(14, 28, 15, [10]),
    ],
)
def test_invalid_counters_rejected(after):
    with pytest.raises(ValueError):
        acceptance(counters(10, 20, 10, [7, 3]), after, 2)


def test_resume_requires_matching_identity_and_unchanged_artifacts(tmp_path):
    artifact = tmp_path / "measurement.json"
    artifact.write_text("raw requests")
    identity = {"model": "a", "repeat": 0}
    save(
        tmp_path / "complete.json",
        {
            "identity": identity,
            "attempt": ".",
            "artifacts": {"measurement.json": digest(artifact)},
        },
    )
    assert completed(tmp_path, identity) == tmp_path
    with pytest.raises(ValueError, match="identity mismatch"):
        completed(tmp_path, {"model": "b"})
    artifact.write_text("truncated")
    with pytest.raises(ValueError, match="changed"):
        completed(tmp_path, identity)


def test_partial_requests_do_not_count_as_complete(tmp_path):
    artifact = tmp_path / "measurement.json"
    artifact.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {
                        "requests": {
                            "successful": [{"output_metrics": {"text_tokens": 3}}]
                        }
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="Expected 2"):
        validate_requests(artifact, 2)


def test_same_count_but_wrong_prompts_is_not_complete(tmp_path):
    artifact = tmp_path / "measurement.json"
    request = {
        "output_metrics": {"text_tokens": 3},
        "request_args": json.dumps(
            {"body": {"messages": [{"role": "user", "content": "wrong"}]}}
        ),
    }
    artifact.write_text(
        json.dumps({"benchmarks": [{"requests": {"successful": [request]}}]})
    )
    with pytest.raises(ValueError, match="prompts differ"):
        validate_requests(artifact, 1, ["original prompt"])


def test_target_only_point_never_requires_speculative_counters(tmp_path, monkeypatch):

    data = tmp_path / "data.jsonl"
    data.write_text(json.dumps({"turns": "original prompt"}) + "\n")
    identity = tmp_path / "run.json"
    save(identity, {"arm": "target-only"})
    args = Namespace(
        identity=identity,
        dataset=str(data),
        max_concurrency=1,
        warmup_requests=2,
        max_tokens=4096,
        eval_seed=0,
        target_only=True,
        output_dir=str(tmp_path / "point"),
        target="http://unused/v1",
    )
    calls = []

    def fake_run(args, dataset, count, destination):
        calls.append(count)
        result = {"metrics": {"output_tokens": count}}
        save(destination, result)
        return result

    monkeypatch.setattr(point, "guidellm", fake_run)
    monkeypatch.setattr(
        point,
        "fetch_metrics",
        lambda *_: pytest.fail("target-only scraped spec counters"),
    )
    point.run_point(args)
    marker = json.loads((tmp_path / "point/complete.json").read_text())
    result = json.loads(
        (tmp_path / "point" / marker["attempt"] / "result.json").read_text()
    )
    assert result["acceptance"] is None
    assert result["acceptance_scope"] == "not_applicable"
    point.run_point(args)
    assert calls == [2, 1]  # A valid resume did not repeat warm-up or measurement.


def test_zero_events_keep_undefined_ratios_missing():
    snapshot = counters(0, 0, 0, [0, 0])
    result = acceptance(snapshot, snapshot, 2)
    assert result["acceptance_length"] is None
    assert result["accepted_token_fraction"] is None
    assert result["per_position_acceptance"] == [None, None]


def test_missing_counters_are_distinct_from_zero_events():
    with pytest.raises(ValueError, match="Missing speculative counters"):
        acceptance("", "", 2)
