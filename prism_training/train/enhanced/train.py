from funlib.geometry import Coordinate
from prism_training.train.losses import WeightedMSELoss
from prism_training.train.models import ChannelAgnosticModel
from prism_training.train.nodes import CreateDiff
import gunpowder as gp
import logging
import torch

logging.basicConfig(level=logging.INFO)

# torch.backends.cudnn.benchmark = True


def train(iterations, create_diff=True, checkpoint_basename=""):
    logging.info(
        f"training for {iterations} iterations using checkpoint base: {checkpoint_basename}"
    )

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

    # assuming this was downloaded
    sample = "../../data/instance/example_data.zarr"

    source = gp.ZarrSource(
        sample,
        {
            raw_key: "raw",
            target_key: "avg_barcodes",
            mask_key: "unlabelled",
            dummy_key: "unlabelled",
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

    pipeline = source

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
        checkpoint_basename=checkpoint_basename,
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
        "--create_diff",
        type=bool,
        default=True,
        help="Whether to compute the barcode difference",
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
        create_diff=args.create_diff,
        checkpoint_basename=args.checkpoint_basename,
    )
