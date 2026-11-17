"""Download published GeoPose checkpoints into a new, labelled directory."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.request
import uuid
import zipfile


ROOT = Path(__file__).resolve().parents[1]
URLS = {
    5: "https://www.dropbox.com/s/0uplf7b5d502ul2/folds_TimmUnet_tf_efficientnetv2_l_in21k_5_r2?dl=1",
    6: "https://www.dropbox.com/s/54dhmlbdip23fif/folds_TimmUnet_tf_efficientnetv2_l_in21k_6_r2?dl=1",
}
EXPECTED_SHA256 = {
    5: "e89a7584c9e59ed5424235aebeed40b50947836d6410a3459027fd0880abb590",
    6: "b4a63c2ad0b7c8446d179028fe50aa93d5c962303df999937ad8939ef6ab29dd",
}


def download(folds, destination):
    destination.mkdir(parents=True, exist_ok=False)
    manifest = {"source": "GeoPose 1st Place, published final checkpoints (not ImageNet initialization)",
                "downloaded_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    for fold in folds:
        name = f"folds_TimmUnet_tf_efficientnetv2_l_in21k_{fold}_r2"
        output = destination / name
        partial = output.with_suffix(".part")
        digest = hashlib.sha256()
        size = 0
        print(f"Downloading author's fold {fold} to {output}", flush=True)
        request = urllib.request.Request(URLS[fold], headers={"User-Agent": "GeoPose-checkpoint-downloader/1.0"})
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("xb") as stream:
            expected = response.headers.get("Content-Length")
            while True:
                chunk = response.read(4 * 1024 * 1024)
                if not chunk:
                    break
                stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            if expected and size != int(expected):
                raise OSError(f"Incomplete checkpoint: {size} of {expected} bytes")
        with partial.open("rb") as stream:
            signature = stream.read(2)
        if digest.hexdigest() != EXPECTED_SHA256[fold]:
            raise ValueError(f"SHA256 mismatch for author's fold {fold}; leaving {partial} unaccepted")
        if size < 1024 * 1024 or signature not in (b"PK", b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"):
            raise ValueError(f"Downloaded file is not a checkpoint: {partial}")
        if signature == b"PK":
            with zipfile.ZipFile(partial) as archive:
                bad = archive.testzip()
                if bad:
                    raise ValueError(f"Corrupt checkpoint archive member: {bad}")
        partial.rename(output)
        manifest["files"][str(fold)] = {"filename": name, "url": URLS[fold],
                                        "bytes": size, "sha256": digest.hexdigest()}
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"Verified file container; {size} bytes; SHA256 {digest.hexdigest()}", flush=True)
    return destination


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", nargs="+", type=int, choices=(5, 6), default=[5, 6])
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    output_dir = args.output_dir or ROOT / "weights/author" / stamp
    print(download(sorted(set(args.folds)), output_dir).resolve())
