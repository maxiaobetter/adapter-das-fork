# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Compare a real GSM8K request alone or beside a real prompt above DSA top-k."""
import argparse
import json
from pathlib import Path

from workload import Service, write


def first_change(a, b):
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                min(len(a), len(b)) if len(a) != len(b) else None)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--dataset-reference", type=Path, required=True)
    p.add_argument("--call-reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument('--require-deterministic', action='store_true')
    args = p.parse_args()
    match = json.loads(args.dataset_reference.read_text())["rows"]
    row = next(r for r in match if r["dataset_index"] == 51)
    anchor = row["response"]["prompt_token_ids"]
    companion = json.loads(args.call_reference.read_text())["prefill_decode_long"]["input_ids"]
    assert len(anchor) < 2048 < len(companion) and isinstance(companion[0], int)
    server = Service(f"http://127.0.0.1:{args.port}")
    cases = {
        "solo": ([anchor], 0),
        "long_companion": ([anchor, companion], 0),
        "long_companion_first": ([companion, anchor], 1),
        "solo_repeat": ([anchor], 0),
    }
    evidence = {}
    expected = None
    expected_counts = None
    count_names = ("spec_verify_ct", "spec_num_correct_drafts", "spec_num_proposed_drafts")
    for label, (prompts, index) in cases.items():
        server.flush()
        with server.post("/generate", {
            "input_ids": prompts,
            "return_prompt_token_ids": True,
            "sampling_params": [{"temperature": 0, "sampling_seed": 42,
                                 "max_new_tokens": 512 if i == index else 8}
                                for i in range(len(prompts))],
        }) as reply:
            actual = json.load(reply)
        assert len(actual) == len(prompts) and not any(r["meta_info"]["cached_tokens"] for r in actual)
        r = actual[index]
        assert r["prompt_token_ids"] == anchor
        counts = tuple(r["meta_info"].get(name) for name in count_names)
        if expected is None:
            expected = r["output_ids"]
            expected_counts = counts
        evidence[label] = {"batch_input_lengths": list(map(len, prompts)),
                           "first_change": first_change(expected, r["output_ids"]),
                           "mtp_counts_equal": None not in counts and counts == expected_counts,
                           "result": r}
        write(args.output, evidence)
        print(label, "first change", evidence[label]["first_change"],
              "MTP counts equal", evidence[label]["mtp_counts_equal"], flush=True)
    if args.require_deterministic and any(
        row["first_change"] is not None or not row["mtp_counts_equal"]
        for row in evidence.values()
    ):
        raise SystemExit("Cross-threshold output tokens or MTP counters differ; see the output JSON")


if __name__ == "__main__":
    main()
