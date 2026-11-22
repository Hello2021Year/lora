#!/usr/bin/env python3
"""Score Swift inference JSONL and verify coverage against the specified holdout."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from prepare_data import LABELS, canonical_hash


def rows(path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            yield json.loads(line)


def example_key(row):
    return canonical_hash([message["content"] for message in row["messages"][:2]])


def candidate_name(row):
    return json.loads(row["messages"][1]["content"])["candidate_tool"]["name"]


def metrics(pairs):
    counts = Counter(pairs)
    n = sum(counts.values())
    recall, f1 = {}, {}
    for label in LABELS:
        correct = counts[label, label]
        actual = sum(v for (gold, _), v in counts.items() if gold == label)
        predicted = sum(v for (_, pred), v in counts.items() if pred == label)
        recall[label] = correct / actual if actual else None
        f1[label] = 2 * correct / (actual + predicted) if actual + predicted else None
    return {
        "rows": n,
        "accuracy": sum(counts[label, label] for label in LABELS) / n,
        "format_pass": 1 - sum(counts[label, "INVALID"] for label in LABELS) / n,
        "macro_f1": sum(f1.values()) / len(LABELS) if None not in f1.values() else None,
        "recall": recall,
        "confusion": {
            gold: {pred: counts[gold, pred] for pred in (*LABELS, "INVALID")} for gold in LABELS
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    args = parser.parse_args()

    by_tool = defaultdict(Counter)
    train_labels = Counter()
    for row in rows(args.train):
        gold = row["messages"][-1]["content"]
        by_tool[candidate_name(row)][gold] += 1
        train_labels[gold] += 1
    majority = max(LABELS, key=train_labels.__getitem__)

    expected = {}
    for row in rows(args.expected):
        key = example_key(row)
        gold = row["messages"][-1]["content"]
        if key in expected or gold not in LABELS:
            parser.error("holdout has duplicate prompts or invalid gold labels")
        expected[key] = (gold, candidate_name(row))

    pairs = []
    seen = set()
    for row in rows(args.predictions):
        key = example_key(row)
        if key in seen or key not in expected:
            parser.error("predictions contain duplicate or unexpected prompts")
        seen.add(key)
        gold, _ = expected[key]
        if row.get("labels") != gold:
            parser.error("prediction label differs from the holdout")
        response = row.get("response")
        pairs.append((gold, response if response in LABELS else "INVALID"))
    if not expected or len(seen) != len(expected):
        parser.error(f"expected {len(expected)} predictions, got {len(seen)}")

    baselines = []
    for gold, tool in expected.values():
        label = max(LABELS, key=by_tool[tool].__getitem__) if tool in by_tool else majority
        baselines.append((gold, label))
    print(json.dumps({
        "model": metrics(pairs),
        "always_majority": metrics((gold, majority) for gold, _ in expected.values()),
        "candidate_name_majority": metrics(baselines),
    }, indent=2))


if __name__ == "__main__":
    main()
