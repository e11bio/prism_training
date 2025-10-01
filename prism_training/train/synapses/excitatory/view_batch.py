import argparse
import neuroglancer
import numpy as np
import zarr
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument(
    "-b",
    "--bind-address",
    type=str,
    default="localhost",
    help="bind address",
)

parser.add_argument(
    "-s",
    "--snapshot",
    type=str,
    default="",
    help="path to snapshot",
)

args = parser.parse_args()

neuroglancer.set_server_bind_address(args.bind_address)

snapshot = args.snapshot

assert Path(snapshot).is_dir() and Path(snapshot).suffix == ".zarr", "Not a zarr"

f = zarr.open(snapshot)

raw = f["raw"]
target = f["target"]
prediction = f["prediction"]

dims = neuroglancer.CoordinateSpace(
    units="nm", names=["c^", "z", "y", "x"], scales=[1] + raw.attrs["voxel_size"]
)

viewer = neuroglancer.Viewer()

shader = """
void main() {
emitRGB(
    vec3(
        toNormalized(getDataValue(0)),
        toNormalized(getDataValue(1)),
        toNormalized(getDataValue()))
    );
}
"""

with viewer.txn() as s:
    for ds_name, data in [
        ("raw", raw),
        ("target", target),
        ("prediction", prediction),
    ]:
        voxel_offset = [
            i // j for i, j in zip(data.attrs["offset"], data.attrs["voxel_size"])
        ]

        # batch size of 8 so just view first batch for simplicity and load
        # into numpy
        data = data[0]

        vol = neuroglancer.LocalVolume(
            data=data, dimensions=dims, voxel_offset=[0] + voxel_offset
        )

        s.layers[ds_name] = neuroglancer.ImageLayer(source=vol, shader=shader)

print(viewer)
