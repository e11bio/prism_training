from botocore import UNSIGNED
from botocore.client import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import boto3
import gdown
import os
import shutil
import tempfile
import zipfile


def _download_one(s3_client, bucket_name, key, dest_path):
    dirname = os.path.dirname(dest_path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    s3_client.download_file(bucket_name, key, dest_path)
    return key


def downloadDirectory(bucket_name, prefix, workers=32, output=None):
    cfg = Config(signature_version=UNSIGNED)
    s3_client = boto3.client("s3", config=cfg)
    s3_resource = boto3.resource("s3", config=cfg)
    bucket = s3_resource.Bucket(bucket_name)

    keys = [
        o.key for o in bucket.objects.filter(Prefix=prefix) if not o.key.endswith("/")
    ]
    if not keys:
        print("No files found.")
        return

    def dest_for(key: str) -> str:
        if output:
            rel = key[len(prefix) :].lstrip("/")
            return os.path.join(output, rel)
        else:
            return key

    print(f"Downloading {len(keys)} files with {workers} workers...")
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [
            ex.submit(_download_one, s3_client, bucket_name, k, dest_for(k))
            for k in keys
        ]
        for f in as_completed(futures):
            k = f.result()
            print(f"Downloaded {k}")


def extract_gdrive_zip(id: str, out_dir: str, flatten_top_level: bool = False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        zip_path = tmp / "tmp.zip"
        extract_dir = tmp / "extracted"
        extract_dir.mkdir()

        # download
        gdown.download(id=id, output=str(zip_path), quiet=False)

        # extract
        with zipfile.ZipFile(zip_path, "r") as zf:
            for m in zf.infolist():
                # skip macOS metadata
                if m.filename.startswith("__MACOSX/") or m.filename.endswith(
                    ".DS_Store"
                ):
                    continue
                # prevent path traversal
                dest = (extract_dir / m.filename).resolve()
                if not str(dest).startswith(str(extract_dir.resolve())):
                    raise RuntimeError(f"Blocked path: {m.filename}")
                zf.extract(m, extract_dir)

        mac = extract_dir / "__MACOSX"
        if mac.exists():
            shutil.rmtree(mac)

        top = [p for p in extract_dir.iterdir() if p.name not in {"__MACOSX"}]
        single_dir = len(top) == 1 and top[0].is_dir()

        # move into out_dir (optionally flatten)
        if flatten_top_level and single_dir:
            for child in top[0].iterdir():
                shutil.move(str(child), str(out_dir / child.name))
        else:
            for item in top:
                shutil.move(str(item), str(out_dir / item.name))


if __name__ == "__main__":
    if not Path("instance/example_data.zarr").exists():
        downloadDirectory(
            "e11bio-prism",
            "ls/models/training_data/instance/crop_2.zarr",
            workers=32,
            output="instance/example_data.zarr",
        )

    if not Path("semantic/example_data.zarr").exists():
        downloadDirectory(
            "e11bio-prism",
            "ls/models/training_data/semantic/crop_2.zarr",
            workers=32,
            output="semantic/example_data.zarr",
        )

    downloadDirectory(
        "e11bio-prism",
        "ls/models/checkpoints",
        workers=4,
        output="../train",
    )

    # tmp download of gdrive synapse data...
    # once it is added to bucket, download like above

    # training crop
    extract_gdrive_zip(id="1wGC5169C1K4DO2B8Xl-b-s_3shhGOC-M", out_dir="synapses")

    # training checkpoint
    extract_gdrive_zip(
        id="197IIMjBxHUucsbE5QGy_-QvOGdzoH9sB",
        out_dir="../train/synapses",
        flatten_top_level=True,
    )
