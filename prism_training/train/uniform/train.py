import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

import dask

dask.config.set(scheduler="single-threaded")

from funlib.persistence import open_ds, Array
from prism_training.train.models import ChannelAgnosticWrapper, SigmoidWrapper, UNet
from prism_training.train.losses import ContrastiveLoss

from pathlib import Path

from dacapo_toolbox.dataset import iterable_dataset
from dacapo_toolbox.vis.preview import cube

import torch

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


fmaps = 1024
# Create the model:
unet = UNet(
    in_channels=18,
    num_fmaps=fmaps,
    fmap_inc_factor=2,
    downsample_factors=[],
    kernel_size_down=[
        [
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
            (1, 1, 1),
        ],
    ],
)

model = torch.nn.Sequential(
    unet,
    torch.nn.Conv3d(fmaps, 24, kernel_size=1),
)

if Path("model_checkpoint_10000").exists():
    print("Loading model checkpoint...")
    model.load_state_dict(
        torch.load("model_checkpoint_10000", map_location=torch.device("cpu"))[
            "model_state_dict"
        ]
    )


input_shape = (10, 212, 212)
output_shape = (10, 212, 212)

# load the data
sample = Path("../../data/instance/example_data.zarr")
raw = open_ds(sample / "enhanced")
labels = open_ds(sample / "labels")


dataset = iterable_dataset(
    datasets={
        "raw": raw,
        "labels": labels,
    },
    shapes={
        "raw": input_shape,
        "labels": output_shape,
    },
    transforms={
        "raw": lambda x: x.float() / 255,
        "labels": lambda x: x.long(),
    },
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_func = ContrastiveLoss(
    t=2,
    a=1,
    regularize=True,
    background_relabel=False,
    uniform_emb=True,
    dynamic_balancing=False,
    batchwise=True,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
for iteration, batch in enumerate(dataloader):
    if iteration == NUM_ITERATIONS:
        break

    raw, labels = (
        batch["raw"].to(device),
        batch["labels"].to(device),
    )

    optimizer.zero_grad()
    pred = model(raw)
    loss_val = loss_func(pred, labels)
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
        print(pred.shape, raw.shape, labels.shape)
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
                "Pred": "pca",
            },
            filename=SNAPSHOT_DIR / f"{iteration}.jpg",
            title="Prediction",
        )
