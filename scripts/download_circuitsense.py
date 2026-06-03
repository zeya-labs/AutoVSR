#!/usr/bin/env python3
"""Download the CircuitSense dataset snapshot from Hugging Face."""

import argparse
import os
import time
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default="armanakbari4/CircuitSense")
    parser.add_argument("--output-dir", type=Path, default=Path("data/CircuitSense"))
    parser.add_argument("--revision", default="main")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--endpoint", default=os.getenv("HF_ENDPOINT"), help="Optional Hugging Face endpoint or mirror, e.g. https://hf-mirror.com")
    parser.add_argument("--token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3, help="Number of whole-snapshot retry attempts after transient download failures.")
    parser.add_argument("--retry-sleep", type=float, default=60.0, help="Seconds to sleep between retry attempts.")
    parser.add_argument("--allow-pattern", action="append", dest="allow_patterns")
    parser.add_argument("--ignore-pattern", action="append", dest="ignore_patterns")
    args = parser.parse_args()

    attempts = max(1, args.retries + 1)
    last_error: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            path = snapshot_download(
                repo_id=args.repo_id,
                repo_type="dataset",
                revision=args.revision,
                cache_dir=args.cache_dir,
                local_dir=args.output_dir,
                endpoint=args.endpoint,
                token=args.token,
                local_files_only=args.local_files_only,
                max_workers=args.max_workers,
                allow_patterns=args.allow_patterns,
                ignore_patterns=args.ignore_patterns,
            )
            break
        except Exception as exc:
            last_error = exc
            if attempt >= attempts:
                raise
            print(f"Download attempt {attempt}/{attempts} failed: {exc}")
            print(f"Sleeping {args.retry_sleep:.1f}s before retrying...")
            time.sleep(args.retry_sleep)
    else:
        raise RuntimeError("Download failed") from last_error
    print(path)


if __name__ == "__main__":
    main()
