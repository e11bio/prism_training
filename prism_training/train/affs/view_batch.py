import argparse
import neuroglancer
import numpy as np
import zarr
from pathlib import Path


def additive_shader(
    num_channels,
    total_default=1.0,
    total_min=0.5,
    total_max=2.5,
    channel_default=1.0,
    channel_min=0.5,
    channel_max=2.5,
):

    slider_controls = (
        "#uicontrol float total slider(default={}, min={}, max={}, step=0.1)".format(
            total_default, total_min, total_max
        )
    )

    r_terms, g_terms, b_terms = [], [], []
    channel_indices = list(range(num_channels))

    for i in channel_indices:
        current_channel_default = channel_default
        current_channel_min = channel_min
        current_channel_max = channel_max

        slider_controls += (
            "\n#uicontrol float channel_"
            + str(i)
            + " slider(default={}, min={}, max={}, step=0.1)".format(
                current_channel_default, current_channel_min, current_channel_max
            )
        )

        term = "channel_" + str(i) + " * toNormalized(getDataValue(" + str(i) + "))"
        if i % 3 == 0:
            r_terms.append(term)
        elif i % 3 == 1:
            g_terms.append(term)
        else:
            b_terms.append(term)

    r_sum = " + ".join(r_terms)
    g_sum = " + ".join(g_terms)
    b_sum = " + ".join(b_terms)

    shader_code = """
{}
void main() {{
    emitRGB(
        total * vec3(
            {},
            {},
            {}
        )
    );
}}
""".format(
        slider_controls, r_sum, g_sum, b_sum
    )

    return shader_code


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
pred_affs = f["pred_affs"]
pred_lsds = f["pred_lsds"]

dims = neuroglancer.CoordinateSpace(
    units="nm", names=["c^", "z", "y", "x"], scales=[1] + raw.attrs["voxel_size"]
)

viewer = neuroglancer.Viewer()

with viewer.txn() as s:
    for ds_name, data in [
        ("raw", raw),
        ("pred_affs", pred_affs),
        ("pred_lsds", pred_lsds),
    ]:
        voxel_offset = [
            i // j for i, j in zip(data.attrs["offset"], data.attrs["voxel_size"])
        ]

        vol = neuroglancer.LocalVolume(
            data=data, dimensions=dims, voxel_offset=[0] + voxel_offset
        )

        if ds_name == "raw":
            shader = additive_shader(data.shape[0])
        else:
            shader = """
void main() {
emitRGB(
    vec3(
        toNormalized(getDataValue(0)),
        toNormalized(getDataValue(1)),
        toNormalized(getDataValue(2)))
    );
}
"""

        s.layers[ds_name] = neuroglancer.ImageLayer(source=vol, shader=shader)

print(viewer)
