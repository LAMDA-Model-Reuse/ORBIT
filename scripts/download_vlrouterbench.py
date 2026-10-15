"""Download and safely unpack the frozen raw release; never execute pickles.

Upstream Steps 1/2 are a separate, explicit preparation step. This helper does
not install/run upstream code or evaluate VLMs. See docs/vl_routerbench.md.
"""

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

REVISION = "6e44fbf50e170f8e72734b49f63f2b52a8866377"
ARCHIVE_SHA256 = "8b52f86e8f6679c300dfd12212f7c6a520840d9906afc1587e887039587dfddc"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/vl_routerbench",
                        help="New or empty preparation directory.")
    parser.add_argument("--archive", type=Path, help="Previously downloaded official archive (hash checked).")
    args = parser.parse_args()
    directory = args.output_dir.resolve()
    if directory.exists() and {p.name for p in directory.iterdir()} - {"hub"}:
        parser.error("--output-dir must be new/empty or contain only a resumable hub cache; existing data will not be overwritten.")
    directory.mkdir(parents=True, exist_ok=True)
    archive_path = args.archive
    if archive_path is None:
        from huggingface_hub import hf_hub_download
        archive_path = Path(hf_hub_download(
            repo_id="KinghtH/VL-RouterBench", repo_type="dataset", revision=REVISION,
            filename="vlm_router_data.tar.gz", cache_dir=str(directory / "hub"),
        ))
    digest = hashlib.sha256()
    with archive_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != ARCHIVE_SHA256:
        raise ValueError("Archive hash does not match the frozen official release; nothing extracted.")
    from utils.data import safe_extract_tar
    with tarfile.open(archive_path, "r:gz") as archive:
        members = archive.getmembers()
        expanded = sum(member.size for member in members if member.isfile())
        if shutil.disk_usage(directory).free < expanded + 256 * 1024 * 1024:
            raise OSError(f"Insufficient disk space for {expanded} bytes of extracted data.")
        safe_extract_tar(archive, str(directory))
    print(f"Verified and safely extracted the raw release to {directory}.")
    print("Next: run upstream Steps 1 and 2 as documented in docs/vl_routerbench.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
