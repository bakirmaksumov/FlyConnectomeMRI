"""Download the original Cheng et al. brain tumor dataset from Figshare (public, CC BY 4.0).

Usage:
    python scripts/download_figshare.py
    python scripts/download_figshare.py --article-id 1512427 --dest data/raw/figshare_1512427

Every file is verified against Figshare's published MD5. The zip archives are
kept as downloaded; extraction is a separate step (scripts/extract_figshare.py).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from flyconnectome_mri.datasets.figshare_download import download_article
from flyconnectome_mri.utils.logging import setup_logging
from flyconnectome_mri.utils.provenance import PROJECT_ROOT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download a public Figshare article with MD5 verification.")
    parser.add_argument("--article-id", type=int, default=1512427)
    parser.add_argument("--dest", default="data/raw/figshare_1512427/original")
    args = parser.parse_args(argv)

    setup_logging()
    dest = Path(args.dest)
    if not dest.is_absolute():
        dest = PROJECT_ROOT / dest
    provenance = download_article(args.article_id, dest)
    print(f"{provenance['title']} (doi {provenance['doi']}, version {provenance['version']}, "
          f"license {provenance['license']})")
    for f in provenance["files"]:
        print(f"  {f['name']:<40} md5 {f['md5']} verified")


if __name__ == "__main__":
    main()
