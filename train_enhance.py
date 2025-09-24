from e11_train.models.channel_agnostic_model import ChannelAgnosticModel
from funlib.geometry import Coordinate
from loss import WeightedMSELoss
from pathlib import Path
import gunpowder as gp
import json
import logging
import math
import numpy as np
import os
import torch

logging.basicConfig(level=logging.INFO)

torch.backends.cudnn.benchmark = True


def save_model(
    model,
    checkpoint_dir,
    min_input_shape,
    min_output_shape,
    min_shape_step,
    in_channels,
    out_channels,
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    meta_data = {
        "min_input_shape": min_input_shape,
        "min_output_shape": min_output_shape,
        "min_shape_step": min_shape_step,
        "in_channels": in_channels,
        "out_channels": out_channels,
    }

    model = model.to("cuda")

    # traced model
    with torch.no_grad():
        m = torch.jit.trace(model, torch.rand(1, 1, *min_input_shape).to("cuda"))
        torch.jit.save(m, f"{checkpoint_dir}/enhance_jit.pt")

    # regular model
    torch.save(model, f"{checkpoint_dir}/model")

    with open(f"{checkpoint_dir}/meta.json", "w") as meta_file:
        json.dump(meta_data, meta_file)


class CreateDiff(gp.BatchFilter):
    def __init__(self, raw_key, target_key, mask_key):
        self.raw_key = raw_key
        self.target_key = target_key
        self.mask_key = mask_key

    def crop_raw(self, raw_data, target_data):
        # lazy crop - forgot how to do this with gunpowder rois

        raw_shape = raw_data.shape[1:]
        target_shape = target_data.shape[1:]

        raw_center_z = raw_shape[0] // 2
        raw_center_y = raw_shape[1] // 2
        raw_center_x = raw_shape[2] // 2

        target_center_z = target_shape[0] // 2
        target_center_y = target_shape[1] // 2
        target_center_x = target_shape[2] // 2

        start_z = raw_center_z - target_center_z
        start_y = raw_center_y - target_center_y
        start_x = raw_center_x - target_center_x
        end_z = start_z + target_shape[0]
        end_y = start_y + target_shape[1]
        end_x = start_x + target_shape[2]

        raw_data = raw_data[:, start_z:end_z, start_y:end_y, start_x:end_x]

        return raw_data

    def process(self, batch, request):
        raw_data = batch[self.raw_key].data
        target_data = batch[self.target_key].data
        mask_data = batch[self.mask_key].data.astype(bool)

        raw_data = self.crop_raw(raw_data, target_data)
        target_data[:, mask_data] -= raw_data[:, mask_data]

        batch[self.target_key].data = target_data


def train(iterations, save_dir, create_diff=False):
    raw_key = gp.ArrayKey("RAW")
    target_key = gp.ArrayKey("TARGET")
    mask_key = gp.ArrayKey("MASK")
    dummy_key = gp.ArrayKey("DUMMY")
    predicted_key = gp.ArrayKey("PREDICTED")

    num_fmaps = 12
    fmap_inc_factor = 5
    upsample_mode = "trilinear"
    use_residual = True

    ds_fact = [(1, 2, 2), (1, 2, 2), (2, 2, 2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3, 3, 3), (3, 3, 3)]] * num_levels
    ksu = [[(3, 3, 3), (3, 3, 3)]] * (num_levels - 1)

    model = ChannelAgnosticModel(
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu,
        upsample_mode=upsample_mode,
        use_residual=use_residual,
    )

    input_shape = [52, 148, 148]
    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[1:]

    print(output_shape)

    checkpoint_dir = f"{save_dir}/checkpoint_data"
    os.makedirs(checkpoint_dir, exist_ok=True)

  #   save_model(
        # model,
        # checkpoint_dir,
        # min_input_shape,
        # min_output_shape,
        # min_shape_step,
        # in_channels=None,
        # out_channels=None,
    # )

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(lr=0.5e-4, params=model.parameters())

    voxel_size = gp.Coordinate((400, 168, 168))

    input_size = gp.Coordinate(input_shape) * voxel_size
    output_size = gp.Coordinate(output_shape) * voxel_size

    request = gp.BatchRequest()
    request.add(raw_key, input_size)
    request.add(target_key, output_size)
    request.add(mask_key, output_size)
    request.add(predicted_key, output_size)
    request.add(dummy_key, voxel_size * 2)

    base = Path("/home/arlo/Desktop/Workspace/Projects/lonely_santa/data/training")

    samples = [
        "lonely_santa/sparse/sample_0.zarr",
        "lonely_santa/sparse/sample_2.zarr",
        "lonely_santa/sparse/sample_6.zarr",
        "lonely_santa/sparse/sample_7.zarr",
        "lonely_santa/sparse/sample_8.zarr",
        "230701/dense/sample_0.zarr",
        "230701/sparse/sample_0.zarr",
        "230701/sparse/sample_1.zarr",
        "230701/sparse/sample_2.zarr",
        "230701/sparse/sample_3.zarr",
        "230701/sparse/sample_4.zarr",
        "230703/sparse/sample_0.zarr",
        "230703/sparse/sample_1.zarr",
    ]

    # higher prob on denser volumes with spines + thin axons
    # medium prob on volumes with many cell bodies
    # lower prob on very sparse volumes
    probabilities = [
        0.1,
        0.1,
        0.2,
        0.1,
        0.1,
        0.2,
        0.1,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
    ]

    def create_source(sample):

        if "lonely_santa/s" in str(sample):
            barcodes_ds = "expanded_avg_barcodes"
            unlabelled_mask_ds = "combined_expanded_unlabelled_mask"
        else:
            barcodes_ds = "avg_barcodes"
            unlabelled_mask_ds = "combined_unlabelled_mask"

        source = gp.ZarrSource(
            sample,
            {
                raw_key: "raw",
                target_key: barcodes_ds,
                mask_key: unlabelled_mask_ds,
                dummy_key: unlabelled_mask_ds,
            },
            {
                raw_key: gp.ArraySpec(interpolatable=True),
                target_key: gp.ArraySpec(interpolatable=False),
                mask_key: gp.ArraySpec(interpolatable=False),
                dummy_key: gp.ArraySpec(interpolatable=False),
            },
        )

        source += gp.Pad(raw_key, None)
        source += gp.Pad(target_key, None)
        source += gp.Pad(mask_key, None)

        source += gp.Normalize(raw_key)
        source += gp.Normalize(target_key)

        source += gp.RandomLocation(mask=dummy_key, min_masked=1)

        return source

    sources = tuple(create_source(base / sample) for sample in samples)

    pipeline = sources

    pipeline += gp.RandomProvider(probabilities=probabilities)

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    pipeline += gp.DeformAugment(
        control_point_spacing=Coordinate((10,) * 3) * voxel_size[1],
        jitter_sigma=Coordinate(2, 2, 2) * voxel_size[1],
        scale_interval=(0.5, 2.0),
        rotate=True,
        subsample=2,
        p=0.5,
    )

    if create_diff:
        pipeline += CreateDiff(raw_key, target_key, mask_key)

    pipeline += gp.Unsqueeze([mask_key], axis=0)

    batch_size = 1
    pipeline += gp.Stack(batch_size)

    pipeline += gp.PreCache(num_workers=10)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={"x": raw_key},
        outputs={0: predicted_key},
        loss_inputs={
            0: predicted_key,
            1: target_key,
            2: mask_key,
        },
        save_every=10000,
        checkpoint_basename=f"{checkpoint_dir}/model",
    )

    pipeline += gp.Squeeze(
        [
            raw_key,
            target_key,
            predicted_key,
        ]
    )

    pipeline += gp.Snapshot(
        dataset_names={
            raw_key: "raw",
            mask_key: "mask",
            target_key: "target",
            predicted_key: "predicted",
        },
        output_dir=f"{save_dir}/snapshots",
        output_filename="batch_{iteration}.zarr",
        every=500,
    )

    pipeline += gp.PrintProfilingStats(every=100)

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":
    iterations = 200000

    train(iterations, save_dir='diff_only_expanded_labels_3d_more_feats', create_diff=True)
