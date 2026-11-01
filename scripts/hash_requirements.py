"""Attach SHA256 hashes of downloaded distributions to exact-version locks.

Usage: python scripts/hash_requirements.py WHEELHOUSE requirements*.txt
Download with pip download --no-deps -r requirements-dev.txt -d WHEELHOUSE first.
These locks target the platform used for the download (Linux x86_64 / Python 3.12).
"""
import argparse
from hashlib import sha256
from pathlib import Path
import re

from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheelhouse", type=Path)
    parser.add_argument("locks", type=Path, nargs="+")
    args = parser.parse_args()
    hashes = {}
    for archive in args.wheelhouse.iterdir():
        if archive.suffix == ".whl":
            name, version, *_ = parse_wheel_filename(archive.name)
        elif archive.name.endswith((".tar.gz", ".zip")):
            name, version = parse_sdist_filename(archive.name)
        else:
            continue
        key = (canonicalize_name(name), str(version))
        hashes.setdefault(key, []).append(sha256(archive.read_bytes()).hexdigest())
    outputs = {}
    for lock in args.locks:
        result = ["# Generated exact-version, hash-verified lock for Linux x86_64 / Python 3.12.",
                  "# Regenerate from the matching .in file; see README.md."]
        for line in lock.read_text().splitlines():
            match = re.match(r"^([A-Za-z0-9_.-]+)==([^\s;]+)", line)
            if not match:
                continue
            name, version = match.groups()
            key = (canonicalize_name(name), version)
            if key not in hashes:
                raise ValueError(f"No downloaded distribution for {name}=={version}")
            result.append(f"{name}=={version} " + " ".join(f"--hash=sha256:{h}" for h in sorted(hashes[key])))
        if len(result) == 2:
            raise ValueError(f"No pinned packages found in {lock}")
        outputs[lock] = "\n".join(result) + "\n"
    for lock, content in outputs.items():
        lock.write_text(content)


if __name__ == "__main__":
    main()
