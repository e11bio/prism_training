import argparse
import neuroglancer
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
raw_offset = raw.attrs["offset"]
voxel_size = raw.attrs["voxel_size"]

target = f["target"]
target_offset = [i // j for i, j in zip(target.attrs["offset"], voxel_size)]
target_shape = target.shape[1:]

prediction = f["predicted"][:]

raw_crop = raw[
    :,
    target_offset[0] : target_offset[0] + target_shape[0],
    target_offset[1] : target_offset[1] + target_shape[1],
    target_offset[2] : target_offset[2] + target_shape[2],
]

added = raw_crop + prediction

dims = neuroglancer.CoordinateSpace(
    units="nm", names=["c^", "z", "y", "x"], scales=[1] + voxel_size
)

viewer = neuroglancer.Viewer()

shader = """
void main() {
emitRGB(
    10.0*vec3(
        toNormalized(getDataValue(0)),
        toNormalized(getDataValue(1)),
        toNormalized(getDataValue(2)))
    );
}
"""

with viewer.txn() as s:
    for ds_name, data in [
        ("raw", raw),
        ("target", target),
        ("prediction", prediction),
        ("added", added),
    ]:
        try:
            voxel_offset = [
                i // j for i, j in zip(data.attrs["offset"], data.attrs["voxel_size"])
            ]
        except:
            voxel_offset = target_offset

        vol = neuroglancer.LocalVolume(
            data=data, dimensions=dims, voxel_offset=[0] + voxel_offset
        )

        s.layers[ds_name] = neuroglancer.ImageLayer(source=vol, shader=shader)

print(viewer)
