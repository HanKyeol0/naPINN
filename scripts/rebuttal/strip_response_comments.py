#!/usr/bin/env python3
"""Emit submission-ready reviewer responses from the audited drafts.

The drafts in ``rebuttal/responses`` carry HTML comments pointing at the strict
aggregate that backs each claim. Those pointers are what make the drafts
auditable, but OpenReview must not receive them, and the venue forbids links.
This script strips the comments, checks the per-reviewer character limit, and
refuses to emit anything that still contains an unresolved marker or a URL.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


COMMENT = re.compile(r"<!--.*?-->", re.S)
MARKER = re.compile(r"\[\[[^\]]*\]\]")
URL = re.compile(r"https?://|www\.")


def render(source: Path) -> str:
    text = COMMENT.sub("", source.read_text(encoding="utf-8"))
    return re.sub(r"\n{3,}", "\n\n", text).strip() + "\n"


def main(args: argparse.Namespace) -> int:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    failures = []
    for source in sorted(args.source_dir.glob("*.md")):
        body = render(source)
        n = len(body)
        problems = []
        if n > args.limit:
            problems.append(f"exceeds limit by {n - args.limit} characters")
        markers = MARKER.findall(body)
        if markers:
            problems.append(f"unresolved markers: {sorted(set(markers))}")
        if URL.search(body):
            problems.append("contains a URL, which the venue forbids")
        status = "OK" if not problems else "; ".join(problems)
        print(f"{source.name:<12} {n:>6}/{args.limit}  {status}")
        if problems:
            failures.append(source.name)
            continue
        (args.output_dir / f"{source.stem}.txt").write_text(
            body, encoding="utf-8"
        )
    if failures:
        print(f"\nNot emitted: {', '.join(failures)}", file=sys.stderr)
        return 1
    print(f"\nWrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir", type=Path, default=Path("rebuttal/responses")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("rebuttal/responses/final")
    )
    parser.add_argument("--limit", type=int, default=10000)
    raise SystemExit(main(parser.parse_args()))
