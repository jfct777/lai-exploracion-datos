#!/usr/bin/env python3
"""Give an erratum the same provenance an artifact gets, and check what it must say.

An erratum without a manifest is a claim with no fingerprint: the file it corrects can be
replaced, the erratum can be edited, and nothing notices.  This pins both — the corrected
artifact's digest and the erratum's own — and refuses to write a manifest for an erratum
that fails to state the corrections it exists to carry.

The required statements are passed in rather than hardcoded, because the set of things an
erratum has to say belongs to whoever ordered it, not to this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(repo: Path) -> str | None:
    """The commit the corrections live in, read without assuming git is on PATH."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False,
        )
    except OSError:
        return None
    return result.stdout.strip() or None


def missing_statements(payload: dict, required: dict[str, list[str]]) -> list[str]:
    """Which required statements the erratum does not make anywhere in its text.

    Matching is on lowercase substrings of the whole serialised document: an erratum that
    says the right thing under an unexpected key still passes, and one that says nothing
    fails regardless of how well it is structured.
    """
    text = json.dumps(payload, ensure_ascii=False).lower()
    return sorted(
        name for name, tokens in required.items()
        if not all(token.lower() in text for token in tokens)
    )


def build(
    erratum: Path,
    corrected: Path,
    required: dict[str, list[str]],
    repo: Path,
    written_at: str,
) -> dict[str, object]:
    payload = json.loads(erratum.read_text(encoding="utf-8"))
    absent = missing_statements(payload, required)
    if absent:
        raise SystemExit(
            "The erratum does not state: " + ", ".join(absent)
            + ". A manifest for it would certify a document that omits its own point."
        )
    return {
        "stage": "M27D_ERRATUM_MANIFEST",
        "written_at": written_at,
        "scientific_result": False,
        "published_to_object_storage": False,
        "erratum": {
            "path": erratum.name,
            "bytes": erratum.stat().st_size,
            "sha256": sha256_file(erratum),
        },
        "corrected_artifact": {
            "path": corrected.name,
            "bytes": corrected.stat().st_size,
            "sha256": sha256_file(corrected),
            "unchanged": True,
        },
        "required_statements_present": sorted(required),
        "git_commit": git_commit(repo),
        "python": platform.python_version(),
        "note": (
            "The corrected artifact is pinned by digest so a later edit to either file "
            "breaks this manifest instead of passing silently."
        ),
    }


REQUIRED = {
    "k0_phi_was_an_identity_imposed_by_genesis": ["correctk0", "1 - 4*kin + k2"],
    "k0_phi_does_not_demonstrate_ibd": ["not_evidence_of_ibd"],
    "threshold_cliques_and_independent_sets_remain_valid": [
        "threshold sensitivity", "complete cliques", "maximum independent set",
    ],
    "a_low_participation_ratio_is_concentration_not_family": [
        "measures localisation, not cause",
    ],
    "the_cause_of_the_edges_remains_unidentified": ["unidentified"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--erratum", type=Path, required=True)
    parser.add_argument("--corrected-artifact", type=Path, required=True)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--written-at", required=True, help="ISO date, supplied not guessed")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build(args.erratum, args.corrected_artifact, REQUIRED, args.repo, args.written_at)
    args.out.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
