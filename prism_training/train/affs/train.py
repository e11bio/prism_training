import gunpowder as gp
import logging
import numpy as np
import torch
from lsd.train.gp import AddLocalShapeDescriptor
from funlib.geometry import Coordinate
from prism_training.train.losses import WeightedMSELoss
from prism_training.train.models import MtlsdModel
from prism_training.train.nodes import (
    Blur,
    ChannelWiseIntensityAugment,
    ExpandChannels,
    ShuffleChannels,
    SimpleDefectAugment,
    ZeroChannels,
)

logging.basicConfig(level=logging.INFO)
# torch.backends.cudnn.benchmark = True


def train(iterations, raw_dataset="raw", checkpoint_basename=""):
    logging.info(
        f"training for {iterations} iterations using checkpoint base: {checkpoint_basename}"
    )

    raw = gp.ArrayKey("RAW")
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
        num_fmaps_out=in_channels,
    )

    for m in model.modules():
        if isinstance(m, torch.nn.Conv3d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

    loss = WeightedMSELoss()
    optimizer = torch.optim.Adam(lr=0.5e-4, params=model.parameters())

    input_shape = [76, 196, 196]
    output_shape = model.forward(torch.empty(size=[1, in_channels] + input_shape))[
        0
    ].shape[2:]

    voxel_size = Coordinate((400, 168, 168))

    input_size = Coordinate(input_shape) * voxel_size
    output_size = Coordinate(output_shape) * voxel_size

    sigma = 10 * voxel_size[1]

    request = gp.BatchRequest()
    request.add(raw, input_size)
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

    # assuming this was downloaded
    sample = "../../data/instance/example_data.zarr"

    source = gp.ZarrSource(
        sample,
        {
            raw: raw_dataset,
            labels: "labels",
            unlabelled: "unlabelled",
            dummy: "unlabelled",
        },
        {
            raw: gp.ArraySpec(interpolatable=True),
            labels: gp.ArraySpec(interpolatable=False),
            unlabelled: gp.ArraySpec(interpolatable=False),
            dummy: gp.ArraySpec(interpolatable=False),
        },
    )

    source += gp.Pad(raw, None)
    source += gp.Pad(labels, None)
    source += gp.Pad(unlabelled, None)

    source += gp.Normalize(raw)

    source += gp.RandomLocation(mask=dummy, min_masked=1)

    pipeline = source

    pipeline += ExpandChannels(raw, num_channels=in_channels)

    pipeline += ShuffleChannels(raw)

    pipeline += gp.SimpleAugment(transpose_only=[1, 2])

    pipeline += gp.DeformAugment(
        control_point_spacing=Coordinate((10,) * 3) * voxel_size[1],
        jitter_sigma=Coordinate(2, 2, 2) * voxel_size[1],
        scale_interval=(0.5, 2.0),
        rotate=True,
        subsample=4,
        p=0.7,
    )

    pipeline += Blur(raw, (0.3, 1.2), p=0.3)

    pipeline += ChannelWiseIntensityAugment(
        raw,
        scale_min=0.9,
        scale_max=1.2,
        shift_min=-0.1,
        shift_max=0.2,
        clip=True,
        p=0.8,
    )

    pipeline += SimpleDefectAugment(
        raw,
        prob_missing=0.1,
        prob_low_contrast=0.1,
        contrast_scale=0.5,
        p=0.2,
        axis=0,
    )

    pipeline += ZeroChannels(raw, num_channels=6, p=0.3)

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
        inputs={"x": raw},
        outputs={0: pred_affs, 1: pred_lsds},
        loss_inputs={
            0: pred_affs,
            1: gt_affs,
            2: affs_weights,
            3: pred_lsds,
            4: gt_lsds,
            5: lsds_weights,
        },
        save_every=10000,
        checkpoint_basename=checkpoint_basename,
    )

    pipeline += gp.Squeeze([raw, labels, gt_lsds, gt_affs, pred_lsds, pred_affs])

    pipeline += gp.Snapshot(
        dataset_names={
            raw: "raw",
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

    with gp.build(pipeline):
        for i in range(iterations):
            pipeline.request_batch(request)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Simple training script options")

    parser.add_argument(
        "-i",
        "--iterations",
        type=int,
        default=100000,
        help="Number of iterations (default: 100000)",
    )

    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="raw",
        help="image dataset to input",
    )

    parser.add_argument(
        "-c",
        "--checkpoint_basename",
        type=str,
        default="",
        help="Base name for checkpoints (default: empty string)",
    )

    args = parser.parse_args()

    train(
        iterations=args.iterations,
        raw_dataset=args.dataset,
        checkpoint_basename=args.checkpoint_basename,
    )
