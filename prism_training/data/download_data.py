from botocore import UNSIGNED
from botocore.client import Config
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import boto3
import os


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


if __name__ == "__main__":

    # note this will download ~3gb of data.

    if not Path("instance/example_data.zarr").exists():
        downloadDirectory(
            "e11bio-prism",
            "ls/models/training_data/instance/crop_2.zarr",
            workers=32,
            output="instance/example_data.zarr",
        )

    if not Path("semantic/example_data.zarr").exists():
        # note that semantic crop_0 refers to the same dataset as instance crop_2
        downloadDirectory(
            "e11bio-prism",
            "ls/models/training_data/semantic/crop_0.zarr",
            workers=32,
            output="semantic/example_data.zarr",
        )

    if not Path("synapses/example_data.zarr").exists():
        downloadDirectory(
            "e11bio-prism",
            "ls/models/training_data/synapses/excitatory/crop_0.zarr",
            workers=32,
            output="synapses/example_data.zarr",
        )

    # download all checkpoints. note this will download ~3.5gb of data
    downloadDirectory(
        "e11bio-prism",
        "ls/models/checkpoints",
        workers=4,
        output="../train",
    )

    # can instead download a specific task checkpoint if preferred, eg:
   #  downloadDirectory(
        # "e11bio-prism",
        # "ls/models/checkpoints/enhanced",
        # workers=4,
        # output="../train/enhanced",
    # )
