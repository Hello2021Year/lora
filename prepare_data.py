#!/usr/bin/env python3
"""Audit the pinned JEV dataset and prepare length-bounded Swift JSONL files."""

import argparse
import hashlib
import json
import math
import random
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory

VERSIONS = {
    "v1": ("219df35b1b72694c03556047b7337d6a1a0474e9",
           "f4d651c658f7b79f94e1ed603a07adf9c9ca50d33e670c849ef3236635612c16"),
    "v2": ("38d3e26eb35c2dbd65618c547982abfb693f554b",
           "9df7eab8c1188bf701859250c55ebb1826cdbd414771528025c75af69f3f964a"),
}
LABELS_BY_VERSION = {"v1": ("YES", "NO"), "v2": ("yes", "no")}
MODEL_REVISION = "37561816e66b5b9153b1b64c10968c18c1e6f4a0"
TOKENIZER_URL = f"https://huggingface.co/Qwen/Qwen3.8-27B/resolve/{MODEL_REVISION}/tokenizer.json"
SPLITS = ("train", "validation", "test")
LENGTH_MARGIN = 32


def compact_payload(payload):
    """Keep the complete candidate definition and history, and simplify the catalogue."""
    return {
        **payload,
        "available_tools": [
            {key: tool[key] for key in ("kind", "name", "description")}
            for tool in payload["available_tools"]
        ],
    }


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def canonical_hash(value):
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode()).hexdigest()


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_sha256(path, expected):
    if expected is not None:
        actual = file_sha256(path)
        if actual != expected:
            raise ValueError(f"Unexpected SHA-256 for {path}: {actual}")


def download(url, path, expected_sha256=None):
    if path.is_file():
        verify_sha256(path, expected_sha256)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp_dir:
        temp_file = Path(temp_dir) / path.name
        subprocess.run(
            ["curl", "-fLsS", "--retry", "2", "--max-time", "300", url, "-o", str(temp_file)],
            check=True,
        )
        verify_sha256(temp_file, expected_sha256)
        temp_file.replace(path)


def render_for_length(messages):
    system, user, assistant = (message["content"] for message in messages)
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n{user}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n\n{assistant}<|im_end|>\n"
    )


def lengths_summary(lengths):
    ordered = sorted(lengths)
    return {
        "min": ordered[0],
        **{f"p{p}": ordered[math.ceil(len(ordered) * p / 100) - 1] for p in (50, 90, 95, 99)},
        "max": ordered[-1],
        "total": sum(ordered),
        "counts_at_caps_with_margin": {
            cap: sum(length + LENGTH_MARGIN <= cap for length in ordered)
            for cap in (8192, 16384, 24576, 32768, 49152, 65536, 98304)
        },
    }


def read_batches(source, size=16):
    batch = []
    for line in source:
        batch.append(json.loads(line)["messages"])
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def prepare_split(archive, split, stage, tokenizer, args):
    lengths, selected = [], []
    labels, kept_labels = Counter(), Counter()
    groups = {"full_user": set(), "state": set(), "initial_user": set()}
    with archive.open(f"{split}.jsonl") as source, (stage / f"{split}.jsonl").open(
        "w", encoding="utf-8"
    ) as output:
        for batch in read_batches(source):
            for messages in batch:
                if [item["role"] for item in messages] != ["system", "user", "assistant"]:
                    raise ValueError(f"{split}: unexpected outer message roles")
                label = messages[-1]["content"]
                if label not in LABELS_BY_VERSION[args.data_version]:
                    raise ValueError(f"{split}: unexpected assistant label {label!r}")
                payload = json.loads(messages[1]["content"])
                history = payload["state_messages"]
                first_user = next(item["content"] for item in history if item["role"] == "user")
                for key, value in zip(groups, (payload, history, first_user)):
                    groups[key].add(canonical_hash(value))
                labels[label] += 1
                if args.representation == "compact":
                    messages[1]["content"] = dump(compact_payload(payload))

            encoded = tokenizer.encode_batch(
                [render_for_length(messages) for messages in batch], add_special_tokens=False
            )
            for messages, tokens in zip(batch, encoded):
                length = len(tokens.ids)
                lengths.append(length)
                if length + LENGTH_MARGIN > args.max_length:
                    continue
                if split == "train" or (args.data_version == "v2" and split == "validation"):
                    selected.append(length)
                kept_labels[messages[-1]["content"]] += 1
                output.write(dump({"messages": messages}) + "\n")

    if not kept_labels or not lengths:
        raise ValueError(f"{split}: no rows survive max_length={args.max_length}")
    report = {
        "rows": len(lengths),
        "labels": dict(labels),
        "kept_rows": sum(kept_labels.values()),
        "kept_labels": dict(kept_labels),
        "dropped_overlength": len(lengths) - sum(kept_labels.values()),
        "tokens": lengths_summary(lengths),
        "unique": {key: len(values) for key, values in groups.items()},
    }
    if selected:
        report["smoke_tokens"] = write_smoke(stage, selected, split, args.data_version)
    return report, groups


def write_smoke(stage, selected, split, version):
    longest = sorted(range(len(selected)), key=selected.__getitem__, reverse=True)
    if version == "v2":
        indices = set(longest[:16])
    else:
        indices = set(longest[:8])
        indices.update(random.Random(42).sample(longest[8:], min(16, len(longest[8:]))))
    filename = "smoke.jsonl" if split == "train" else "smoke_validation.jsonl"
    with (stage / f"{split}.jsonl").open(encoding="utf-8") as source, (stage / filename).open(
        "w", encoding="utf-8"
    ) as output:
        for index, line in enumerate(source):
            if index in indices:
                output.write(line)
    return [selected[index] for index in sorted(indices)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-version", choices=VERSIONS, default="v1")
    parser.add_argument("--archive", type=Path)
    parser.add_argument("--tokenizer", type=Path, default=Path("inputs/tokenizer.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--representation", choices=("compact", "full"), default="compact")
    parser.add_argument("--max-length", type=int)
    args = parser.parse_args()
    if args.max_length is None:
        args.max_length = 65536 if args.data_version == "v2" else 49152
    if args.output.exists():
        parser.error(f"choose a new --output directory: {args.output}")
    if args.max_length <= LENGTH_MARGIN:
        parser.error(f"--max-length must exceed {LENGTH_MARGIN}")

    from tokenizers import Tokenizer

    revision, checksum = VERSIONS[args.data_version]
    archive_name = f"jev_datasets_{args.data_version}.zip"
    if args.archive is None:
        args.archive = Path("dataV2" if args.data_version == "v2" else "inputs") / archive_name
    download(f"https://huggingface.co/datasets/sunpengliang/test/resolve/{revision}/{archive_name}",
             args.archive, checksum)
    download(TOKENIZER_URL, args.tokenizer)
    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=args.output.parent) as temp_dir:
        stage = Path(temp_dir) / "prepared"
        stage.mkdir()
        report = {
            "data_version": args.data_version,
            "source_revision": revision,
            "archive_sha256": checksum,
            "reference_model_revision": MODEL_REVISION,
            "tokenizer_sha256": file_sha256(args.tokenizer),
            "representation": args.representation,
            "max_length": args.max_length,
            "length_margin": LENGTH_MARGIN,
            "splits": {},
        }
        groups = {}
        with zipfile.ZipFile(args.archive) as archive:
            for split in SPLITS:
                report["splits"][split], groups[split] = prepare_split(
                    archive, split, stage, tokenizer, args
                )
                print(f"{split}: {report['splits'][split]['kept_rows']} rows kept", flush=True)
        report["split_overlap"] = {
            f"{left}/{right}": {
                key: len(groups[left][key] & groups[right][key]) for key in groups[left]
            }
            for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
        }
        (stage / "audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        stage.rename(args.output)
    print(f"Audit and dataset written to {args.output}")


if __name__ == "__main__":
    main()
