import multiprocessing as mp

if __name__ == "__main__":
    mp.set_start_method("fork", force=True)

import dask

dask.config.set(scheduler="single-threaded")

from prism_training.train.models import ChannelAgnosticWrapper, SigmoidWrapper
from funlib.learn.torch.models import UNet

from pathlib import Path
from dacapo_toolbox.vis.preview import cube

import torch


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

torch.save(model, "pred_model-tmp.pt")

from volara_torch.blockwise import Predict
from volara_torch.models import TorchModel
from volara.datasets import Raw

enhanced = Raw(store="")

checkpoint = TorchModel(in_channels=None, out_channels=1, min_input_shape=...)

semantic_pred = Predict(
    checkpoint=checkpoint,
    in_data=enhanced,
    out_data=[probabilities],
    num_workers=1,
    num_cache_workers=4,
)
