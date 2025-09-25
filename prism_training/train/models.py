import math
import torch
import torch.nn as nn

# workaround - figure out issue with resunet convpass for mtlsd net
from funlib.learn.torch.models import UNet as BaseUNet
from funlib.learn.torch.models import ConvPass as BaseConvPass


class ConvPass(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_sizes,
        activation,
        padding="valid",
        use_residual=False,
    ):
        super(ConvPass, self).__init__()

        if activation is not None:
            activation = getattr(nn, activation)

        self.activation = activation
        layers = []

        if use_residual:
            residual = []

        for index, kernel_size in enumerate(kernel_sizes):
            self.dims = len(kernel_size)

            conv = {
                2: nn.Conv2d,
                3: nn.Conv3d,
            }[self.dims]

            if padding == "same":
                pad = tuple(k // 2 for k in kernel_size)
            else:
                pad = 0

            try:
                layers.append(conv(in_channels, out_channels, kernel_size, padding=pad))
            except KeyError:
                raise RuntimeError("%dD convolution not implemented" % self.dims)

            if use_residual:
                if index == 0:
                    residual.append(
                        conv(in_channels, out_channels, kernel_size=1, padding=pad)
                    )

            in_channels = out_channels

            if activation is not None and index < len(kernel_sizes) - 1:
                layers.append(activation())

        self.out_activation = None

        if activation is not None:
            self.out_activation = activation()

        self.conv_pass = nn.Sequential(*layers)

        if use_residual:
            self.residual = nn.Sequential(*residual)
        else:
            self.residual = None

    def crop(self, to_crop, target_size):
        z, y, x = target_size[-3:]
        sx = (to_crop.shape[-1] - x) // 2
        sy = (to_crop.shape[-2] - y) // 2
        sz = (to_crop.shape[-3] - z) // 2

        return to_crop[..., sz : sz + z, sy : sy + y, sx : sx + x]

    def forward(self, x):
        out = self.conv_pass(x)

        if self.residual is not None:
            res = self.residual(x)

            cropped = self.crop(res, out.size())

            ret = out + cropped

            if self.activation is not None:
                ret = self.out_activation(ret)

            return ret

        else:
            return out


class Downsample(torch.nn.Module):
    def __init__(self, downsample_factor):
        super(Downsample, self).__init__()

        self.dims = len(downsample_factor)
        self.downsample_factor = downsample_factor

        pool = {
            2: torch.nn.MaxPool2d,
            3: torch.nn.MaxPool3d,
            4: torch.nn.MaxPool3d,  # only 3D pooling, even for 4D input
        }[self.dims]

        self.down = pool(downsample_factor, stride=downsample_factor)

    def forward(self, x):
        for d in range(1, self.dims + 1):
            if x.size()[-d] % self.downsample_factor[-d] != 0:
                raise RuntimeError(
                    "Can not downsample shape %s with factor %s, mismatch "
                    "in spatial dimension %d"
                    % (x.size(), self.downsample_factor, self.dims - d)
                )

        return self.down(x)


class Upsample(torch.nn.Module):
    def __init__(
        self,
        scale_factor,
        mode="transposed_conv",
        in_channels=None,
        out_channels=None,
        crop_factor=None,
        next_conv_kernel_sizes=None,
    ):
        super(Upsample, self).__init__()

        assert (crop_factor is None) == (next_conv_kernel_sizes is None), (
            "crop_factor and next_conv_kernel_sizes have to be given together"
        )

        self.crop_factor = crop_factor
        self.next_conv_kernel_sizes = next_conv_kernel_sizes

        self.dims = len(scale_factor)

        if mode == "transposed_conv":
            up = {2: torch.nn.ConvTranspose2d, 3: torch.nn.ConvTranspose3d}[self.dims]

            self.up = up(
                in_channels, out_channels, kernel_size=scale_factor, stride=scale_factor
            )

        else:
            self.up = torch.nn.Upsample(scale_factor=scale_factor, mode=mode)

    def crop_to_factor(self, x, factor, kernel_sizes):
        """Crop feature maps to ensure translation equivariance with stride of
        upsampling factor. This should be done right after upsampling, before
        application of the convolutions with the given kernel sizes.

        The crop could be done after the convolutions, but it is more efficient
        to do that before (feature maps will be smaller).
        """

        shape = x.size()
        spatial_shape = shape[-self.dims :]

        # the crop that will already be done due to the convolutions
        convolution_crop = tuple(
            sum(ks[d] - 1 for ks in kernel_sizes) for d in range(self.dims)
        )

        # we need (spatial_shape - convolution_crop) to be a multiple of
        # factor, i.e.:
        #
        # (s - c) = n*k
        #
        # we want to find the largest n for which s' = n*k + c <= s
        #
        # n = floor((s - c)/k)
        #
        # this gives us the target shape s'
        #
        # s' = n*k + c

        ns = (
            int(math.floor(float(s - c) / f))
            for s, c, f in zip(spatial_shape, convolution_crop, factor)
        )
        target_spatial_shape = tuple(
            n * f + c for n, c, f in zip(ns, convolution_crop, factor)
        )

        if target_spatial_shape != spatial_shape:
            assert all(
                ((t > c) for t, c in zip(target_spatial_shape, convolution_crop))
            ), (
                "Feature map with shape %s is too small to ensure "
                "translation equivariance with factor %s and following "
                "convolutions %s" % (shape, factor, kernel_sizes)
            )

            return self.crop(x, target_spatial_shape)

        return x

    def crop(self, x, shape):
        """Center-crop x to match spatial dimensions given by shape."""

        x_target_size = x.size()[: -self.dims] + shape

        offset = tuple((a - b) // 2 for a, b in zip(x.size(), x_target_size))

        slices = tuple(slice(o, o + s) for o, s in zip(offset, x_target_size))

        return x[slices]

    def forward(self, f_left, g_out):
        g_up = self.up(g_out)

        if self.next_conv_kernel_sizes is not None:
            g_cropped = self.crop_to_factor(
                g_up, self.crop_factor, self.next_conv_kernel_sizes
            )
        else:
            g_cropped = g_up

        f_cropped = self.crop(f_left, g_cropped.size()[-self.dims :])

        return torch.cat([f_cropped, g_cropped], dim=1)


class UNet(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down=None,
        kernel_size_up=None,
        activation="ReLU",
        fov=(1, 1, 1),
        voxel_size=(1, 1, 1),
        num_fmaps_out=None,
        num_heads=1,
        upsample_mode="nearest",
        padding="valid",
        use_residual=False,
    ):
        """Create a U-Net::

            f_in --> f_left --------------------------->> f_right--> f_out
                        |                                   ^
                        v                                   |
                     g_in --> g_left ------->> g_right --> g_out
                                 |               ^
                                 v               |
                                       ...

        where each ``-->`` is a convolution pass, each `-->>` a crop, and down
        and up arrows are max-pooling and transposed convolutions,
        respectively.

        The U-Net expects 3D or 4D tensors shaped like::

            ``(batch=1, channels, [length,] depth, height, width)``.

        This U-Net performs only "valid" convolutions, i.e., sizes of the
        feature maps decrease after each convolution. It will perfrom 4D
        convolutions as long as ``length`` is greater than 1. As soon as
        ``length`` is 1 due to a valid convolution, the time dimension will be
        dropped and tensors with ``(b, c, z, y, x)`` will be use (and returned)
        from there on.

        Args:

            in_channels:

                The number of input channels.

            num_fmaps:

                The number of feature maps in the first layer. This is also the
                number of output feature maps. Stored in the ``channels``
                dimension.

            fmap_inc_factor:

                By how much to multiply the number of feature maps between
                layers. If layer 0 has ``k`` feature maps, layer ``l`` will
                have ``k*fmap_inc_factor**l``.

            downsample_factors:

                List of tuples ``(z, y, x)`` to use to down- and up-sample the
                feature maps between layers.

            kernel_size_down (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the left side. Kernel sizes
                can be given as tuples or integer. If not given, each
                convolutional pass will consist of two 3x3x3 convolutions.

            kernel_size_up (optional):

                List of lists of kernel sizes. The number of sizes in a list
                determines the number of convolutional layers in the
                corresponding level of the build on the right side. Within one
                of the lists going from left to right. Kernel sizes can be
                given as tuples or integer. If not given, each convolutional
                pass will consist of two 3x3x3 convolutions.

            activation:

                Which activation to use after a convolution. Accepts the name
                of any tensorflow activation function (e.g., ``ReLU`` for
                ``torch.nn.ReLU``).

            fov (optional):

                Initial field of view in physical units

            voxel_size (optional):

                Size of a voxel in the input data, in physical units

            num_heads (optional):

                Number of decoders. The resulting U-Net has one single encoder
                path and num_heads decoder paths. This is useful in a
                multi-task learning context.

            constant_upsample (optional):

                If set to true, perform a constant upsampling instead of a
                transposed convolution in the upsampling layers.

            padding (optional):

                How to pad convolutions. Either 'same' or 'valid' (default).
        """

        super(UNet, self).__init__()

        self.num_levels = len(downsample_factors) + 1
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.out_channels = num_fmaps_out if num_fmaps_out else num_fmaps

        # default arguments

        if kernel_size_down is None:
            kernel_size_down = [[(3, 3, 3), (3, 3, 3)]] * self.num_levels
        if kernel_size_up is None:
            kernel_size_up = [[(3, 3, 3), (3, 3, 3)]] * (self.num_levels - 1)

        # compute crop factors for translation equivariance
        crop_factors = []
        factor_product = None
        for factor in downsample_factors[::-1]:
            if factor_product is None:
                factor_product = list(factor)
            else:
                factor_product = list(f * ff for f, ff in zip(factor, factor_product))
            crop_factors.append(factor_product)
        crop_factors = crop_factors[::-1]

        # modules

        # left convolutional passes
        self.l_conv = nn.ModuleList(
            [
                ConvPass(
                    (
                        in_channels
                        if level == 0
                        else num_fmaps * fmap_inc_factor ** (level - 1)
                    ),
                    num_fmaps * fmap_inc_factor**level,
                    kernel_size_down[level],
                    activation=activation,
                    padding=padding,
                    use_residual=use_residual,
                )
                for level in range(self.num_levels)
            ]
        )
        self.dims = self.l_conv[0].dims

        # left downsample layers
        self.l_down = nn.ModuleList(
            [
                Downsample(downsample_factors[level])
                for level in range(self.num_levels - 1)
            ]
        )

        # right up/crop/concatenate layers
        self.r_up = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        Upsample(
                            downsample_factors[level],
                            mode=upsample_mode,
                            in_channels=num_fmaps * fmap_inc_factor ** (level + 1),
                            out_channels=num_fmaps * fmap_inc_factor ** (level + 1),
                            crop_factor=(
                                crop_factors[level] if padding != "same" else None
                            ),
                            next_conv_kernel_sizes=(
                                kernel_size_up[level] if padding != "same" else None
                            ),
                        )
                        for level in range(self.num_levels - 1)
                    ]
                )
                for _ in range(num_heads)
            ]
        )

        # right convolutional passes
        self.r_conv = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        ConvPass(
                            num_fmaps * fmap_inc_factor**level
                            + num_fmaps * fmap_inc_factor ** (level + 1),
                            (
                                num_fmaps * fmap_inc_factor**level
                                if num_fmaps_out is None or level != 0
                                else num_fmaps_out
                            ),
                            kernel_size_up[level],
                            activation=activation,
                            padding=padding,
                            use_residual=use_residual,
                        )
                        for level in range(self.num_levels - 1)
                    ]
                )
                for _ in range(num_heads)
            ]
        )

        # print(self.num_levels)

    def rec_forward(self, level, f_in):
        # index of level in layer arrays
        i = self.num_levels - level - 1

        # convolve
        f_left = self.l_conv[i](f_in)

        # end of recursion
        if level == 0:
            fs_out = [f_left] * self.num_heads

        else:
            # down
            g_in = self.l_down[i](f_left)

            # nested levels
            gs_out = self.rec_forward(level - 1, g_in)

            # up, concat, and crop
            fs_right = [
                self.r_up[h][i](f_left, gs_out[h]) for h in range(self.num_heads)
            ]

            # convolve
            fs_out = [self.r_conv[h][i](fs_right[h]) for h in range(self.num_heads)]

        return fs_out

    def forward(self, x):
        y = self.rec_forward(self.num_levels - 1, x)

        if self.num_heads == 1:
            return y[0]

        return y


class ChannelAgnosticModel(torch.nn.Module):
    def __init__(
        self,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        upsample_mode,
        use_residual,
        iter_channels=False,
        add_residual=False,
    ):
        super().__init__()

        # UNet with in_channels=1 since we are processing one channel at a time
        self.unet = UNet(
            in_channels=1,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            upsample_mode=upsample_mode,
            use_residual=use_residual,
        )

        # ConvPass layer to get the final denoised output for each channel
        self.head = ConvPass(num_fmaps, 1, [[1, 1, 1]], activation=None)
        self.iter_channels = iter_channels
        self.add_residual = add_residual

    def forward(self, x):
        B, C, z_in, y_in, x_in = x.shape

        if self.iter_channels:
            channel_outputs = []

            for c in range(C):
                single_channel_batch = x[:, c, :, :, :]
                single_channel_batch = single_channel_batch.unsqueeze(1)
                processed_channel_batch = self.unet(single_channel_batch)
                processed_channel_batch = self.head(processed_channel_batch)

                channel_outputs.append(processed_channel_batch)

            output = torch.cat(channel_outputs, dim=1)

            _, _, z_out, y_out, x_out = output.shape

        else:
            x = x.view(B * C, 1, z_in, y_in, x_in)
            y = self.unet(x)
            y = self.head(y)
            _, _, z_out, y_out, x_out = y.shape

            output = y.view(B, C, z_out, y_out, x_out)

        if self.add_residual:
            z_start = (z_in - z_out) // 2
            y_start = (y_in - y_out) // 2
            x_start = (x_in - x_out) // 2

            x = x[
                :,
                :,
                z_start : z_start + z_out,
                y_start : y_start + y_out,
                x_start : x_start + x_out,
            ]

            output = output + x

        return output


class MtlsdModel(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        num_fmaps,
        fmap_inc_factor,
        downsample_factors,
        kernel_size_down,
        kernel_size_up,
        constant_upsample,
        num_heads=1,
        num_lsds=10,
        num_affs=3,
        num_fmaps_out=None,
    ):
        super().__init__()

        self.num_heads = num_heads

        # create unet
        self.unet = BaseUNet(
            in_channels=in_channels,
            num_fmaps=num_fmaps,
            fmap_inc_factor=fmap_inc_factor,
            downsample_factors=downsample_factors,
            kernel_size_down=kernel_size_down,
            kernel_size_up=kernel_size_up,
            constant_upsample=constant_upsample,
            num_heads=num_heads,
            num_fmaps_out=num_fmaps_out if num_fmaps_out is not None else num_fmaps,
        )

        final_conv_kernel = [[1] * len(downsample_factors[0])]

        # create affs and lsds heads
        self.aff_head = BaseConvPass(
            num_fmaps_out, num_affs, final_conv_kernel, activation="Sigmoid"
        )
        self.lsd_head = BaseConvPass(
            num_fmaps_out, num_lsds, final_conv_kernel, activation="Sigmoid"
        )

    def forward(self, x):
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        # pass raw through unet
        z = self.unet(x)

        # pass output through heads. todo: handle extra heads
        if self.num_heads == 1:
            affs = self.aff_head(z)
            lsds = self.lsd_head(z)
        else:
            affs = self.aff_head(z[1])
            lsds = self.lsd_head(z[0])

        return affs, lsds


class ChannelAgnosticWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        B, C, Z, Y, X = x.shape

        # reshape batch dim
        x = x.reshape(B * C, 1, Z, Y, X)

        if self.training:
            x = self.model(x)
        else:
            # During inference we iterate over channels to reduce memory usage
            outs = []
            for i in range(C):
                outs.append(self.model(x[i * B : (i + 1) * B]))
            x = torch.cat(outs, dim=0)

        # different output shape from valid convs
        _, OUT_C, Z, Y, X = x.shape

        # reshape back and take the max accross input channels
        x, _ = x.reshape(B, C, OUT_C, Z, Y, X).max(dim=1)

        if OUT_C == 1:
            x = x.squeeze(1)

        return x


class SigmoidWrapper(torch.nn.Module):
    """
    Sigmoid wrapper makes it easier to train with a logits loss function
    since you want to pass logits to `BCEWithLogitsLoss` during training
    but generate probabilities with `sigmoid` during inference.
    """

    def __init__(self, model):
        super().__init__()
        self.add_module("0", model)

    def forward(self, x):
        x = self._modules["0"](x)
        if self.training:
            return x
        else:
            return torch.sigmoid(x)
