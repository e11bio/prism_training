import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

import dask
import dask.array as da

dask.config.set(scheduler="single-threaded")

from funlib.persistence import open_ds, Array
from prism_training.train.models import ChannelAgnosticWrapper, SigmoidWrapper
from funlib.learn.torch.models import UNet

from pathlib import Path

from dacapo_toolbox.dataset import (
    iterable_dataset,
    DeformAugmentConfig,
    SimpleAugmentConfig,
    MaskedSampling,
)
from dacapo_toolbox.vis.preview import cube

import torch
import torchvision

# Training parameters:
NUM_ITERATIONS = 1
LOG_INTERVAL = 10
SAVE_CHECKPOINT_EVERY = 1000
SAVE_SNAPSHOT_EVERY = 100
SNAPSHOT_DIR = Path("snapshots")
CHECKPOINT_DIR = Path("checkpoints")

# Setup
if not SNAPSHOT_DIR.exists():
    SNAPSHOT_DIR.mkdir(parents=True)
if not CHECKPOINT_DIR.exists():
    CHECKPOINT_DIR.mkdir(parents=True)


# Helper functions
def to_array(data, metadata):
    return Array(
        data,
        offset=metadata[0],
        voxel_size=metadata[1],
    )


fmaps = 12
# Create the model:
unet = UNet(
    in_channels=1,
    num_fmaps=fmaps,
    fmap_inc_factor=2,
    downsample_factors=[
        (1, 2, 2),
        (2, 2, 2),
        (2, 2, 2),
    ],
    kernel_size_down=[
        [(1, 3, 3), (1, 3, 3)],
        [(1, 3, 3), (1, 3, 3)],
        [(3, 3, 3), (3, 3, 3)],
        [(3, 3, 3), (3, 3, 3)],
    ],
    kernel_size_up=[
        [(1, 3, 3), (1, 3, 3)],
        [(1, 3, 3), (1, 3, 3)],
        [(3, 3, 3), (3, 3, 3)],
    ],
    constant_upsample=True,
)

model = SigmoidWrapper(
    ChannelAgnosticWrapper(
        torch.nn.Sequential(
            unet,
            torch.nn.Conv3d(fmaps, 1, kernel_size=1),
        )
    )
)

if Path("model_checkpoint_100000").exists():
    print("Loading model checkpoint...")
    model.load_state_dict(
        torch.load("model_checkpoint_100000", map_location=torch.device("cpu"))[
            "model_state_dict"
        ]
    )


input_shape = (48, 148, 148)
output_shape = (16, 56, 56)

# load the data
# sample = Path("../../data/semantic/example_data.zarr")

# raw = open_ds(sample / "enhanced")
# labels = open_ds(sample / "fgbg")
# mask = open_ds(sample / "border_mask")

raw = open_ds(
    "/home/arlo/Desktop/Workspace/Projects/lonely_santa/data/training/lonely_santa/sparse/sample_6.zarr/enhanced_3d"
)
labels = open_ds(
    "/home/arlo/Desktop/Workspace/Projects/lonely_santa/data/training/lonely_santa/sparse/sample_6.zarr/eroded_labels"
)
mask = open_ds(
    "/home/arlo/Desktop/Workspace/Projects/lonely_santa/data/training/lonely_santa/sparse/sample_6.zarr/unlabeled_mask"
)

dataset = iterable_dataset(
    datasets={
        "raw": raw,
        "labels": labels,
        "mask": mask,
    },
    shapes={
        "raw": input_shape,
        "labels": output_shape,
        "mask": output_shape,
    },
    transforms={
        "raw": lambda x: x.float() / 255,
        "labels": lambda x: x.float(),
        "mask": lambda x: x.float(),
    },
    deform_augment_config=DeformAugmentConfig(
        p=0.5,
        control_point_spacing=(2, 10, 10),
        jitter_sigma=(0.5, 2, 2),
        rotate=True,
        subsample=4,
        rotation_axes=(1, 2),
        scale_interval=(0.9, 1.1),
    ),
    simple_augment_config=SimpleAugmentConfig(
        p=1.0,
        mirror_only=(1, 2),
        transpose_only=(1, 2),
    ),
    sampling_strategies=[MaskedSampling(mask_key="mask", min_masked=0.2)],
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = torchvision.ops.sigmoid_focal_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
for iteration, batch in enumerate(dataloader):
    if iteration == NUM_ITERATIONS:
        break

    raw, labels, mask = (
        batch["raw"].to(device),
        batch["labels"].to(device),
        batch["mask"].to(device),
    )

    optimizer.zero_grad()
    pred = model(raw)
    loss_val = (loss_func(pred, labels) * mask).sum() / mask.sum()
    loss_val.backward()
    optimizer.step()

    if iteration % LOG_INTERVAL == 0:
        print(f"Iteration {iteration}: loss = {loss_val.item()}")

    if iteration % SAVE_SNAPSHOT_EVERY == 0:
        raw_arr = to_array(raw.cpu().detach().numpy()[0], batch["metadata"]["raw"])
        labels_arr = to_array(
            labels.cpu().detach().numpy()[0], batch["metadata"]["labels"]
        )
        raw_arr.lazy_op(labels_arr.roi)
        cube(
            arrays={
                "Raw": raw_arr,
                "Labels": labels_arr,
                "Pred": to_array(
                    torch.nn.functional.sigmoid(pred[0]).cpu().detach().numpy(),
                    batch["metadata"]["labels"],
                ),
            },
            array_types={
                "Raw": "pca",
                "Labels": "labels",
                "Pred": "raw",
            },
            filename=SNAPSHOT_DIR / f"{iteration}.jpg",
            title="Prediction",
        )
