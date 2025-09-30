from funlib.geometry import Coordinate
from pathlib import Path
from prism_training.train.models import ChannelAgnosticModel
from volara.datasets import Raw
from volara.workers import LocalWorker
from volara_torch.blockwise import Predict
from volara_torch.models import TorchModel
import math
import torch


if __name__ == "__main__":
    num_fmaps = 12
    fmap_inc_factor = 5
    upsample_mode = "trilinear"
    use_residual = True

    # iterate over channels rather than treating as batch dim. This allows us to
    # put more data on the gpu
    iter_channels = True

    # add the predicted residual back to the raw data. if training with
    # create_diff=True then we should do this
    add_residual = True

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
        iter_channels=iter_channels,
        add_residual=add_residual,
    )

    input_shape = [52, 148, 148]
    output_shape = model.forward(torch.empty(size=[1, 1] + input_shape))[0].shape[1:]

    min_step_shape = [math.prod(ds_fact) for ds_fact in zip(*ds_fact)]

    torch.save(model, "model_test.pt")

    min_input_shape = [52, 148, 148]
    min_output_shape = [20, 56, 56]
    min_step_shape = [2, 8, 8]

    base = Path("../../data/instance/example_data.zarr")

    raw_dataset = Raw(store=base / "raw", scale_shift=(1 / 255, 0))

    enhanced_dataset = Raw(
        store=base / "enhanced",
    )

    pred_size_growth = Coordinate((8 * 5, 8 * 10, 8 * 10))

    torch_model = TorchModel(
        save_path="model_test.pt",
        checkpoint_file="model_checkpoint_60000",
        in_channels=1,
        out_channels=None,
        min_input_shape=min_input_shape,
        min_output_shape=min_output_shape,
        min_step_shape=min_step_shape,
        out_range=[0, 1],
        pred_size_growth=pred_size_growth,
    )

    predict = Predict(
        checkpoint=torch_model,
        in_data=raw_dataset,
        out_data=[enhanced_dataset],
        num_workers=1,
        num_cache_workers=5,
        worker_config=LocalWorker(),
    )

    # predict.drop()
    # predict.run_blockwise(multiprocessing=False)
    predict.run_blockwise()
