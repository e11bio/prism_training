import gunpowder as gp
import logging
import numpy as np
import os
import json
import torch
import math
import random
import wandb
from lsd.train.gp import AddLocalShapeDescriptor
from funlib.geometry import Coordinate
from e11_train.models.mtlsd_model import MtlsdModel
from e11_train.losses import WeightedMSELoss
from e11_train.gp.nodes import (
    Blur,
    ChannelWiseIntensityAugment,
    ExpandChannels,
    ShuffleChannels,
    SimpleDefectAugment,
    ZeroChannels,
)
from e11_train.utils.wandb import initialize_or_resume_run
from pathlib import Path

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
        m = torch.jit.trace(
            model, torch.rand(1, in_channels, *min_input_shape).to("cuda")
        )
        torch.jit.save(m, f"{checkpoint_dir}/affs_jit.pt")

    # regular model
    torch.save(model, f"{checkpoint_dir}/model")

    with open(f"{checkpoint_dir}/meta.json", "w") as meta_file:
        json.dump(meta_data, meta_file)


class BlendImages(gp.BatchFilter):
    def __init__(self, image_1, image_2, p=1.0):
        self.image_1 = image_1
        self.image_2 = image_2
        self.p = p

    def skip_node(self, request):
        return random.random() > self.p

    def process(self, batch, request):
        image_1_data = batch[self.image_1].data
        image_2_data = batch[self.image_2].data

        weight_a = np.random.rand()
        weight_b = 1 - weight_a

        blended_image = image_1_data * weight_a + image_2_data * weight_b

        batch[self.image_1].data = blended_image


def train(iterations):
    raw = gp.ArrayKey("RAW")
    enhanced = gp.ArrayKey("ENHANCED")
    labels = gp.ArrayKey("LABELS")
    gt_lsds = gp.ArrayKey("GT_LSDS")
    lsds_weights = gp.ArrayKey("LSDS_WEIGHTS")
    pred_lsds = gp.ArrayKey("PRED_LSDS")
    gt_affs = gp.ArrayKey("GT_AFFS")
    gt_affs_mask = gp.ArrayKey("GT_AFFS_MASK")
    affs_weights = gp.ArrayKey("AFFS_WEIGHTS")
    pred_affs = gp.ArrayKey("PRED_AFFS")
    unlabelled = gp.ArrayKey("UNLABELLED")
    dummy = gp.ArrayKey("DUMMY")

    neighborhood = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [6, 0, 0],
            [0, 9, 0],
            [0, 0, 9],
            [12, 0, 0],
            [0, 27, 0],
            [0, 0, 27],
        ]
    )

    in_channels = 18
    num_fmaps = 12
    fmap_inc_factor = 5
    ds_fact = [(1, 2, 2), (1, 2, 2), (2, 2, 2)]
    num_levels = len(ds_fact) + 1
    ksd = [[(3, 3, 3), (3, 3, 3)]] * num_levels
    ksu = [[(3, 3, 3), (3, 3, 3)]] * (num_levels - 1)
    constant_upsample = True

    model = MtlsdModel(
        in_channels=in_channels,
        num_fmaps=num_fmaps,
        fmap_inc_factor=fmap_inc_factor,
        downsample_factors=ds_fact,
        kernel_size_down=ksd,
        kernel_size_up=ksu,
        constant_upsample=constant_upsample,
        num_affs=len(neighborhood),
        num_fmaps_out=in_channels
    )

    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(lr=0.5e-4, params=model.parameters())

    min_input_shape = [36, 100, 100]
    min_output_shape = [4, 8, 8]
    min_shape_step = [math.prod(ds_fact) for ds_fact in zip(*ds_fact)]

    input_shape = [76, 196, 196]
    output_shape = model.forward(torch.empty(size=[1, in_channels] + input_shape))[
        0
    ].shape[2:]

    checkpoint_dir = "checkpoint_data"

   #  save_model(
        # model,
        # checkpoint_dir,
        # min_input_shape,
        # min_output_shape,
        # min_shape_step,
        # in_channels=in_channels,
        # out_channels=[len(neighborhood), 10],
    # )

    print(input_shape, output_shape)

    voxel_size = Coordinate((400, 168, 168))

    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size

    sigma = 10 * voxel_size[1]

    request = gp.BatchRequest()
    request.add(raw, input_size)
    request.add(enhanced, input_size)
    request.add(labels, output_size)
    request.add(unlabelled, output_size)
    request.add(gt_lsds, output_size)
    request.add(lsds_weights, output_size)
    request.add(pred_lsds, output_size)
    request.add(gt_affs, output_size)
    request.add(gt_affs_mask, output_size)
    request.add(affs_weights, output_size)
    request.add(pred_affs, output_size)

    request.add(dummy, voxel_size * 2)

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
        0.15,
        0.15,
        0.15,
        0.15,
        0.15,
        0.1,
        0.1,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
        0.01,
    ]

    def create_source(sample):
        raw_source = gp.ZarrSource(
            sample,
            {raw: "raw", enhanced: "enhanced_3d"},
            {
                raw: gp.ArraySpec(interpolatable=True),
                enhanced: gp.ArraySpec(interpolatable=True),
            },
        )

        labels_source = gp.ZarrSource(
            sample,
            {
                labels: "labels",
                unlabelled: "unlabelled_mask",
                dummy: "unlabelled_mask",
            },
            {
                labels: gp.ArraySpec(interpolatable=False),
                unlabelled: gp.ArraySpec(interpolatable=False),
                dummy: gp.ArraySpec(interpolatable=False),
            },
        )

        raw_source += gp.Normalize(raw)
        raw_source += gp.Normalize(enhanced)

        with gp.build(labels_source):
            roi = labels_source.spec[labels].roi

        raw_source += gp.Crop(raw, roi)
        raw_source += gp.Crop(enhanced, roi)

        source = (raw_source, labels_source) + gp.MergeProvider()

        source += gp.Pad(raw, None)
        source += gp.Pad(enhanced, None)
        source += gp.Pad(labels, None)
        source += gp.Pad(unlabelled, None)

        source += gp.RandomLocation(mask=dummy, min_masked=1)

        return source

    sources = tuple(create_source(base / sample) for sample in samples)

    pipeline = sources

    pipeline += gp.RandomProvider(probabilities=probabilities)

    pipeline += BlendImages(enhanced, raw, p=0.5)

    pipeline += ExpandChannels(enhanced, num_channels=in_channels)

    pipeline += ShuffleChannels(enhanced)

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    pipeline += gp.DeformAugment(
        control_point_spacing=Coordinate((10,) * 3) * voxel_size[1],
        jitter_sigma=Coordinate(2, 2, 2) * voxel_size[1],
        scale_interval=(0.5, 2.0),
        rotate=True,
        subsample=4,
        p=0.7,
    )

    pipeline += Blur(enhanced, (0.3, 1.2), p=0.3)

    pipeline += ChannelWiseIntensityAugment(
        enhanced,
        scale_min=0.9,
        scale_max=1.2,
        shift_min=-0.1,
        shift_max=0.2,
        clip=True,
        p=0.8,
    )

    pipeline += SimpleDefectAugment(
        enhanced,
        prob_missing=0.1,
        prob_low_contrast=0.1,
        contrast_scale=0.5,
        p=0.2,
        axis=0,
    )

    pipeline += ZeroChannels(enhanced, num_channels=6, p=0.3)

    pipeline += AddLocalShapeDescriptor(
        labels,
        gt_lsds,
        lsds_mask=lsds_weights,
        unlabelled=unlabelled,
        sigma=sigma,
        downsample=2,
    )

    pipeline += gp.AddAffinities(
        affinity_neighborhood=neighborhood,
        labels=labels,
        unlabelled=unlabelled,
        affinities=gt_affs,
        affinities_mask=gt_affs_mask,
        dtype=np.float32,
    )

    pipeline += gp.BalanceLabels(
        gt_affs, affs_weights, mask=gt_affs_mask, slab=(3, -1, -1, -1)
    )

    pipeline += gp.Stack(1)

    pipeline += gp.PreCache(num_workers=10)

    pipeline += gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={"x": enhanced},
        outputs={0: pred_affs, 1: pred_lsds},
        loss_inputs={
            0: pred_affs,
            1: gt_affs,
            2: affs_weights,
            3: pred_lsds,
            4: gt_lsds,
            5: lsds_weights,
        },
        checkpoint_basename=checkpoint_dir + "/model",
        save_every=10000,
    )

    pipeline += gp.Squeeze([enhanced, labels, gt_lsds, gt_affs, pred_lsds, pred_affs])

    pipeline += gp.Snapshot(
        dataset_names={
            enhanced: "enhanced",
            labels: "labels",
            unlabelled: "unlabelled",
            gt_lsds: "gt_lsds",
            affs_weights: "affs_weights",
            pred_lsds: "pred_lsds",
            gt_affs: "gt_affs",
            pred_affs: "pred_affs",
        },
        output_filename="batch_{iteration}.zarr",
        every=500,
    )

    pipeline += gp.PrintProfilingStats(every=100)

    initialize_or_resume_run(
        entity="arlo",
        project="lonely_santa",
        run_name="affs_lsds_3d_enhanced",
        resume_run=False,
    )

    with gp.build(pipeline):
        for i in range(iterations):
            batch = pipeline.request_batch(request)

            if (i + 1) % 10:
                wandb.log({"loss": batch.loss, "iteration": batch.iteration})


if __name__ == "__main__":
    train(300000)

    wandb.finish()
