"""Resolve a PhysDox data argument to a local JSON file.

Accepted values for ``--data``:

* ``silver``    Silver set (5,000 items), public.
* ``gold_dev``  Gold dev subset (100 items), public.
* ``gold``      Full Gold set (683 items). Gated: needs ``HF_TOKEN`` of an account with access.
* a path to a local ``.json`` file, or an ``http(s)`` URL.

Downloads are cached under ``~/.cache/physdox`` (override with ``PHYSDOX_CACHE``).
"""

from __future__ import annotations

import os
import urllib.error
import urllib.request
from pathlib import Path

HF_PUBLIC = "phishphish/PhysDox"
HF_GOLD = "phishphish/PhysDox-Gold"

NAMED = {
    "silver": (HF_PUBLIC, "silver/physdox_silver_5000.json"),
    "gold_dev": (HF_PUBLIC, "gold_dev/physdox_gold_dev_100.json"),
    "gold": (HF_GOLD, "gold/physdox_gold_683.json"),
}


def cache_dir() -> Path:
    return Path(os.environ.get("PHYSDOX_CACHE", Path.home() / ".cache" / "physdox"))


def hf_url(repo: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"


def download(url: str, dest: Path, token: str | None = None) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp, tmp.open("wb") as f:
            while chunk := resp.read(1 << 20):
                f.write(chunk)
    except urllib.error.HTTPError as e:
        tmp.unlink(missing_ok=True)
        if e.code in (401, 403):
            raise SystemExit(
                f"Access denied ({e.code}) for {url}\n"
                "The full Gold set is gated. Request access at "
                f"https://huggingface.co/datasets/{HF_GOLD} and set HF_TOKEN to your access token."
            ) from None
        raise
    tmp.replace(dest)
    return dest


def resolve_data(value: str) -> Path:
    if value in NAMED:
        repo, path = NAMED[value]
        dest = cache_dir() / repo.replace("/", "__") / path
        if not dest.exists():
            print(f"Downloading {value} from huggingface.co/datasets/{repo} ...")
            download(hf_url(repo, path), dest, token=os.environ.get("HF_TOKEN"))
        return dest
    if value.startswith(("http://", "https://")):
        dest = cache_dir() / "url" / Path(value.split("?")[0]).name
        if not dest.exists():
            download(value, dest, token=os.environ.get("HF_TOKEN"))
        return dest
    path = Path(value)
    if not path.exists():
        raise SystemExit(f"Data file not found: {value}. Use 'silver', 'gold_dev', 'gold', or a path.")
    return path
