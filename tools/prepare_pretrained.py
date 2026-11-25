#!/usr/bin/env python3
"""Download pinned T5 files and verify the exact pretrained files used by this release.

ActionCLIP must be obtained from its upstream model zoo. This utility verifies
that checkpoint when --actionclip-checkpoint is supplied; it does not substitute
ordinary CLIP weights for the trained ActionCLIP temporal-fusion checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hintnet.checkpoints import ACTIONCLIP_SHA256, T5_MODEL, T5_REVISION

T5_FILES = {
    "config.json": "46dd7cb62d29c81fb551e0ef1ea274c24a46ba441eeb948897706252933df033",
    "model.safetensors": "a90903540cc02cbeb7ff9f823f1a80eb778c7e22426a0e620b01c77a5ec8f5b4",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t5-dir", type=Path, default=Path("pretrained/t5-base"))
    parser.add_argument("--verify-only", action="store_true", help="verify existing files without network access")
    parser.add_argument("--actionclip-checkpoint", type=Path)
    args = parser.parse_args()
    if not args.verify_only:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=T5_MODEL, revision=T5_REVISION,
                          allow_patterns=list(T5_FILES), local_dir=args.t5_dir)
    verified = {}
    for name, expected in T5_FILES.items():
        path = args.t5_dir / name
        if not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Missing or different pinned T5 file: {path}")
        verified[name] = expected
    if args.actionclip_checkpoint:
        if sha256(args.actionclip_checkpoint) != ACTIONCLIP_SHA256:
            raise ValueError("ActionCLIP checkpoint differs from the release ViT-B/16 8-frame source")
        verified["actionclip_vit_b16_8f.pt"] = ACTIONCLIP_SHA256
    print(json.dumps({"status": "pass", "t5_model": T5_MODEL,
                      "t5_revision": T5_REVISION, "verified_sha256": verified}, indent=2))


if __name__ == "__main__":
    main()
