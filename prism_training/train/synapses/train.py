from funlib.learn.torch.models import UNet, ConvPass
from prism_training.train.losses import WeightedBCELossMultiChannel
import gunpowder as gp
import logging
import math
import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
torch.backends.cudnn.benchmark = True


def train(iterations, checkpoint_basename=""):
    logging.info(
        f"training for {iterations} iterations using checkpoint base: {checkpoint_basename}"
    )

    excitatory = gp.ArrayKey("EXCITATORY")

    gt_psd95_output = gp.ArrayKey("GT_BASSOON_OUTPUT")
    prediction = gp.ArrayKey("PREDICTION")
    psd95_mask = gp.ArrayKey("UNLABELED_MASK")
    psd95_mask_weight = gp.ArrayKey("MASK_WEIGHT")
    psd95_mask_underseg = gp.ArrayKey("MASK_UNDERSEG")

    num_fmaps = 12
    unet = UNet(
        in_channels=2,
        num_fmaps=num_fmaps,
        fmap_inc_factor=5,
        downsample_factors=[(2,) * 3, (2,) * 3],
        padding="valid",
        constant_upsample=True,
    )

    model = torch.nn.Sequential(
        unet,
        ConvPass(
            in_channels=num_fmaps,
            out_channels=2,
            kernel_sizes=[(1, 1, 1)],
            activation="Sigmoid",
        ),
    )

    loss = WeightedBCELossMultiChannel()
    optimizer = torch.optim.Adam(lr=0.5e-4, params=model.parameters())

    voxel_size = gp.Coordinate([400, 168, 168])
    input_shape = gp.Coordinate([80, 80, 80])
    output_shape = gp.Coordinate([40, 40, 40])  # ?
    input_size = input_shape * voxel_size
    output_size = output_shape * voxel_size

    # assuming it was downloaded...
    samples = ["../../data/synapses/crop1_excitatory.zarr"]

    data_sources = tuple(
        gp.ZarrSource(
            s,
            datasets={
                excitatory: "volumes/ls_2channels_globalnorm",
                gt_psd95_output: "volumes/psd95_output",
                psd95_mask: "volumes/psd95_mask",
                psd95_mask_weight: "volumes/psd95_weights",
                psd95_mask_underseg: "volumes/psd95_underseg",
            },
            array_specs={
                excitatory: gp.ArraySpec(interpolatable=True),
                gt_psd95_output: gp.ArraySpec(interpolatable=False),
                psd95_mask: gp.ArraySpec(interpolatable=False),
                psd95_mask_weight: gp.ArraySpec(interpolatable=False),
                psd95_mask_underseg: gp.ArraySpec(interpolatable=False),
            },
        )
        + gp.Normalize(excitatory)
        + gp.RandomLocation()
        + gp.Reject(mask=psd95_mask_underseg, min_masked=0.001, reject_probability=0.9)
        for s in samples
    )

    train = gp.torch.Train(
        model,
        loss,
        optimizer,
        inputs={"input": excitatory},
        loss_inputs={0: prediction, 1: gt_psd95_output, 2: psd95_mask_weight},
        outputs={0: prediction},
        save_every=5000,
        log_dir="log",
        checkpoint_basename=checkpoint_basename,
    )

    pipeline = (
        data_sources
        + gp.RandomProvider()
        + gp.SimpleAugment(transpose_only=[1, 2], mirror_only=[1, 2])
        + gp.IntensityAugment(excitatory, 0.8, 1.2, -0.2, 0.2)
        + gp.Stack(8)
        + train
        + gp.Snapshot(
            dataset_names={
                excitatory: "raw",
                gt_psd95_output: "target",
                prediction: "prediction",
            },
            output_filename="batch_{iteration}.zarr",
            every=100,
        )
    )

    request = gp.BatchRequest()
    request.add(excitatory, input_size)
    request.add(psd95_mask, output_size)
    request.add(gt_psd95_output, output_size)
    request.add(psd95_mask_weight, output_size)
    request.add(psd95_mask_underseg, output_size)
    request.add(prediction, output_size)

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
        "-c",
        "--checkpoint_basename",
        type=str,
        default="",
        help="Base name for checkpoints (default: empty string)",
    )

    args = parser.parse_args()

    train(iterations=args.iterations, checkpoint_basename=args.checkpoint_basename)
