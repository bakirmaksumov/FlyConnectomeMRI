"""Extract the Figshare zip archives into one flat folder of read-only .mat files.

Usage:
    python scripts/extract_figshare.py
    python scripts/extract_figshare.py --archives data/raw/figshare_1512427/original --output data/raw/figshare_1512427/mat

The downloaded archives are not modified (they are also made read-only). Running
this again verifies already-extracted files against the archives instead of
overwriting them. A table of every extracted file with its source archive and
SHA-256 is written to <output>/extracted_files.csv.
"""

from __future__ import annotations

import argparse
import os
import stat
from pathlib import Path

from flyconnectome_mri.datasets.figshare import extract_archives
from flyconnectome_mri.utils.logging import setup_logging
from flyconnectome_mri.utils.provenance import PROJECT_ROOT


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Extract Figshare archives (read-only, verified).")
    parser.add_argument("--archives", default="data/raw/figshare_1512427/original")
    parser.add_argument("--output", default="data/raw/figshare_1512427/mat")
    args = parser.parse_args(argv)

    logger = setup_logging()
    archives = Path(args.archives) if Path(args.archives).is_absolute() else PROJECT_ROOT / args.archives
    output = Path(args.output) if Path(args.output).is_absolute() else PROJECT_ROOT / args.output

    for original in archives.iterdir():
        if original.is_file() and not original.name.endswith(".part"):
            os.chmod(original, stat.S_IREAD)  # protect the downloaded originals
    table = extract_archives(archives, output)
    table.to_csv(output / "extracted_files.csv", index=False, lineterminator="\n")
    logger.info("%d .mat files extracted/verified in %s", len(table), output)
    print(table.groupby("archive").size().to_string())


if __name__ == "__main__":
    main()
