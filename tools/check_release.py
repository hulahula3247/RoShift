"""Verify exported file hashes and optionally scan externally supplied identifiers."""

import argparse
import hashlib
import json
import re
from pathlib import Path
import zipfile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--deny-file", type=Path, help="Private newline-separated identifiers; keep this file outside the release")
    args = parser.parse_args()
    root = args.root.resolve()
    manifest = json.loads((root/"SHA256SUMS.json").read_text(encoding="utf-8"))
    deny = [x.strip().casefold() for x in args.deny_file.read_text(encoding="utf-8").splitlines() if x.strip()] if args.deny_file else []
    bad = []
    private = []
    credentials=[]
    secret_pattern=re.compile(r'\bhf_[A-Za-z0-9]{25,}\b|\bAKIA[A-Z0-9]{16}\b|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----')
    for name, expected in manifest.items():
        path = (root/name).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            bad.append(name)
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            bad.append(name)
        texts = [name, raw.decode("utf-8", errors="replace")]
        if path.suffix == ".pt" and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as z:
                texts.extend(z.namelist())
                texts.extend(z.read(n).decode("utf-8", errors="replace") for n in z.namelist() if n.endswith(".pkl"))
        if any(marker in text.casefold() for text in texts for marker in deny):
            private.append(name)
        if any(secret_pattern.search(text) for text in texts):
            credentials.append(name)
    print(json.dumps({"files_checked": len(manifest), "hash_failures": bad,
                      "identifier_matches": private, "credential_pattern_matches":credentials,
                      "identifier_list_supplied": bool(deny)}, indent=2))
    raise SystemExit(1 if bad or private or credentials else 0)


if __name__ == "__main__":
    main()
