"""Download a public Figshare article with checksum verification and provenance.

Only public files are fetched (no credentials). Each file is downloaded to
``<name>.part``, resumed with an HTTP Range request if the connection drops,
verified against the MD5 that Figshare publishes, and only then renamed to its
final name. Files that already exist with the correct MD5 are not downloaded again.

The article's API record is saved as ``figshare_article.json`` and a summary
(URLs, checksums, license, download date) as ``download_provenance.json``.
"""

from __future__ import annotations

import hashlib
import json
import time
import urllib.request
from pathlib import Path
from typing import Any

from flyconnectome_mri.utils.logging import get_logger
from flyconnectome_mri.utils.provenance import utc_now_iso

API_URL = "https://api.figshare.com/v2/articles/{article_id}"
_CHUNK = 1 << 20
_MAX_ATTEMPTS = 8


class ChecksumError(RuntimeError):
    """Raised when a downloaded file does not match its published MD5."""


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_article(article_id: int) -> dict[str, Any]:
    """Return Figshare's public JSON record for ``article_id``."""
    with urllib.request.urlopen(API_URL.format(article_id=article_id), timeout=60) as response:
        return json.load(response)


def download_file(url: str, dest: Path, expected_md5: str, expected_size: int | None = None) -> None:
    """Download ``url`` to ``dest`` (resumable), then verify its MD5.

    Raises:
        ChecksumError: If the finished file's MD5 differs from ``expected_md5``.
    """
    logger = get_logger()
    if dest.exists() and md5_file(dest) == expected_md5:
        logger.info("Already present and verified: %s", dest.name)
        return
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, _MAX_ATTEMPTS + 1):
        offset = part.stat().st_size if part.exists() else 0
        if expected_size is not None and offset >= expected_size:
            break
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                # 206 = the server honoured the Range header; otherwise start over.
                mode = "ab" if offset and response.status == 206 else "wb"
                with open(part, mode) as f:
                    for chunk in iter(lambda: response.read(_CHUNK), b""):
                        f.write(chunk)
            break
        except OSError as exc:  # network errors (URLError, timeouts, resets) are OSErrors
            logger.warning("Download of %s interrupted (%s), attempt %d/%d; resuming",
                           dest.name, type(exc).__name__, attempt, _MAX_ATTEMPTS)
            time.sleep(min(2 ** attempt, 30))
    else:
        raise RuntimeError(f"Could not download {url} after {_MAX_ATTEMPTS} attempts")

    actual = md5_file(part)
    if actual != expected_md5:
        part.unlink()
        raise ChecksumError(f"{dest.name}: MD5 {actual} does not match published {expected_md5}; file removed")
    part.replace(dest)
    logger.info("Downloaded and verified: %s", dest.name)


def download_article(article_id: int, dest_dir: Path, article: dict[str, Any] | None = None) -> dict[str, Any]:
    """Download every file of a public Figshare article into ``dest_dir``; returns the provenance dict."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    article = article if article is not None else fetch_article(article_id)
    (dest_dir / "figshare_article.json").write_text(json.dumps(article, indent=2), encoding="utf-8")

    files = []
    for entry in article["files"]:
        dest = dest_dir / entry["name"]
        download_file(entry["download_url"], dest, entry["computed_md5"], entry.get("size"))
        files.append({"name": entry["name"], "url": entry["download_url"], "size": entry.get("size"),
                      "md5": entry["computed_md5"], "md5_verified": True})

    provenance = {
        "article_id": article_id,
        "title": article.get("title"),
        "doi": article.get("doi"),
        "version": article.get("version"),
        "published_date": article.get("published_date"),
        "authors": [a.get("full_name") for a in article.get("authors", [])],
        "license": (article.get("license") or {}).get("name"),
        "license_url": (article.get("license") or {}).get("url"),
        "api_url": API_URL.format(article_id=article_id),
        "downloaded_utc": utc_now_iso(),
        "files": files,
    }
    (dest_dir / "download_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    return provenance
