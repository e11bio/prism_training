from scipy.ndimage import gaussian_filter
from skimage.draw import ellipsoid
import gunpowder as gp
import logging
import numpy as np
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Blur(gp.BatchFilter):
    def __init__(self, array, blur_range, p=1.0):
        self.array = array
        self.blur_range = blur_range
        self.p = p

    def skip_node(self, request):
        return random.random() > self.p

    def process(self, batch, request):
        array = batch[self.array].data

        # different numbers will simulate noisier or cleaner array
        sigma = random.uniform(self.blur_range[0], self.blur_range[1])

        for z in range(array.shape[1]):
            array_sec = array[:, z]

            array[:, z] = np.array(
                [
                    gaussian_filter(array_sec[i], sigma=sigma)
                    for i in range(array_sec.shape[0])
                ]
            ).astype(array_sec.dtype)

        batch[self.array].data = array


class ChannelWiseIntensityAugment(gp.IntensityAugment):
    # assumes 4d raw data of c,z,y,x or 5d raw data of b,c,z,y,x
    def __init__(
        self,
        array,
        scale_min,
        scale_max,
        shift_min,
        shift_max,
        z_section_wise=False,
        clip=True,
        p=1.0,
        batch_dim: bool = False,
    ):
        super().__init__(
            array,
            scale_min,
            scale_max,
            shift_min,
            shift_max,
            z_section_wise,
            clip,
        )
        self.batch_dim = batch_dim

    def skip_node(self, request):
        return random.random() > self.p

    def process(self, batch, request):
        raw = batch.arrays[self.array]

        if self.batch_dim:
            try:
                data = raw.data.transpose(1, 0, *range(2, raw.data.ndim))
            except ValueError as e:
                raise ValueError(raw.data.shape) from e
        else:
            data = raw.data

        assert data.dtype == np.float32 or data.dtype == np.float64, (
            "Intensity augmentation requires float types for the raw array (not "
            + str(data.dtype)
            + "). Consider using Normalize before."
        )
        if self.clip:
            assert (
                data.min() >= 0 and data.max() <= 1
            ), "Intensity augmentation expects raw values in [0,1]. Consider \
                    using Normalize before."

        for c in range(data.shape[0]):
            data[c] = self.__augment(
                data[c],
                np.random.uniform(low=self.scale_min, high=self.scale_max),
                np.random.uniform(low=self.shift_min, high=self.shift_max),
            )

        # clip values, we might have pushed them out of [0,1]
        if self.clip:
            data[data > 1] = 1
            data[data < 0] = 0

        if self.batch_dim:
            raw.data = data.transpose(1, 0, *range(2, data.ndim))
        else:
            raw.data = data

    def __augment(self, a, scale, shift):
        return a.mean() + (a - a.mean()) * scale + shift


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


class ExpandChannels(gp.BatchFilter):
    def __init__(self, array, num_channels):
        self.array = array
        self.num_channels = num_channels

    def process(self, batch, request):
        data = batch.arrays[self.array].data

        assert len(data.shape) == 4, "Data must be 4-dimensional."

        # Get the current number of channels
        current_channels = data.shape[0]

        # Check if the desired number of channels is less than current channels
        if self.num_channels < current_channels:
            raise ValueError(
                "num_channels should be greater than the current number of channels."
            )

        # do nothing if we don't need to expand
        if self.num_channels == current_channels:
            return

        # Compute how many channels to duplicate
        duplicate_channels = self.num_channels - current_channels

        # Randomly select channels to duplicate
        channels_to_duplicate = np.random.choice(current_channels, duplicate_channels)

        # Concatenate the original channels and the duplicated channels
        expanded = np.concatenate([data, data[channels_to_duplicate]], axis=0)

        # write the data to the batch
        batch[self.array].data = expanded


class ShuffleChannels(gp.BatchFilter):
    def __init__(self, array, p=1.0, batch_dim: bool = False):
        self.array = array
        self.p = p
        self.batch_dim = batch_dim

    def skip_node(self, request):
        return random.random() > self.p

    def process(self, batch, request):
        array = batch.arrays[self.array]

        if self.batch_dim:
            data = array.data.transpose(1, 0, *range(2, array.data.ndim))
        else:
            data = array.data

        num_channels = data.shape[0]

        channel_perm = np.random.permutation(num_channels)

        data = data[channel_perm]

        if self.batch_dim:
            data = data.transpose(1, 0, *range(2, data.ndim))

        array.data = data


class SimpleDefectAugment(gp.BatchFilter):
    """Augment intensity arrays section-wise with artifacts like missing
    sections, low-contrast sections.

    Args:

        intensities (:class:`ArrayKey`):

            The key of the array of intensities to modify.

        prob_missing(``float``):
        prob_low_contrast(``float``):

        contrast_scale (``float``, optional):

            By how much to scale the intensities for a low-contrast section,
            used if ``prob_low_contrast`` > 0.

        axis (``int``, optional):

            Along which axis sections are cut.
    """

    def __init__(
        self,
        intensities,
        prob_missing=0.05,
        prob_low_contrast=0.05,
        contrast_scale=0.1,
        axis=0,
        p=1.0,
    ):
        self.intensities = intensities
        self.prob_missing = prob_missing
        self.prob_low_contrast = prob_low_contrast
        self.contrast_scale = contrast_scale
        self.axis = axis
        self.p = p

    def skip_node(self, request):
        return random.random() > self.p

    # send roi request to data-source upstream
    def prepare(self, request):
        deps = gp.BatchRequest()

        # we prepare the augmentations, by determining which slices
        # will be augmented by which method

        prob_missing_threshold = self.prob_missing
        prob_low_contrast_threshold = prob_missing_threshold + self.prob_low_contrast

        spec = request[self.intensities].copy()
        roi = spec.roi
        logging.debug("downstream request ROI is %s" % roi)
        raw_voxel_size = self.spec[self.intensities].voxel_size

        # store the mapping slice to augmentation type in a dict
        self.slice_to_augmentation = {}
        for c in range((roi / raw_voxel_size).shape[self.axis]):
            r = random.random()

            if r < prob_missing_threshold:
                logging.debug("Zero-out " + str(c))
                self.slice_to_augmentation[c] = "zero_out"

            elif r < prob_low_contrast_threshold:
                logging.debug("Lower contrast " + str(c))
                self.slice_to_augmentation[c] = "lower_contrast"

        deps[self.intensities] = spec

    def process(self, batch, request):
        # assert batch.get_total_roi().dims == 3, "defectaugment works on 3d batches only"

        raw = batch.arrays[self.intensities]

        for c, augmentation_type in self.slice_to_augmentation.items():
            section_selector = tuple(
                slice(None if d != self.axis else c, None if d != self.axis else c + 1)
                for d in range(raw.spec.roi.dims)
            )

            section_selector = (slice(None),) + section_selector

            if augmentation_type == "zero_out":
                raw.data[section_selector] = 0

            elif augmentation_type == "low_contrast":
                section = raw.data[section_selector]

                mean = section.mean()
                section -= mean
                section *= self.contrast_scale
                section += mean

                raw.data[:, section_selector] = section


class ZeroChannels(gp.BatchFilter):
    def __init__(self, array, num_channels=0, p=1.0):
        self.array = array
        self.num_channels = num_channels
        self.p = p

    def skip_node(self, request):
        return random.random() > self.p

    def get_bounds(self, size):
        start, end = (
            np.random.randint(0, size // 3),
            np.random.randint(2 * size // 3, size),
        )

        return start, end

    def draw_random_shape(self, z_size, y_size, x_size, shape_type="ellipsoid"):
        # Define random start and end points
        start_z, end_z = self.get_bounds(z_size)
        start_y, end_y = self.get_bounds(y_size)
        start_x, end_x = self.get_bounds(x_size)

        if shape_type == "ellipsoid":
            z_radius = (end_z - start_z) // 2
            y_radius = (end_y - start_y) // 2
            x_radius = (end_x - start_x) // 2

            ellipsoid_volume = ellipsoid(x_radius, y_radius, z_radius)
            zz, yy, xx = np.where(ellipsoid_volume)

            zz += start_z + z_radius
            yy += start_y + y_radius
            xx += start_x + x_radius

        elif shape_type == "rectangle":
            zz, yy, xx = np.meshgrid(
                np.arange(start_z, end_z),
                np.arange(start_y, end_y),
                np.arange(start_x, end_x),
                indexing="ij",
            )

        elif shape_type == "points":
            max_radius = random.randint(3, 10)  # todo: parameterize
            k = random.randint(10, 100)  # todo: parameterize
            zz, yy, xx = [], [], []

            for _ in range(k):
                center_z = np.random.randint(max_radius, z_size - max_radius)
                center_y = np.random.randint(max_radius, y_size - max_radius)
                center_x = np.random.randint(max_radius, x_size - max_radius)

                # Randomly determine the size of the ellipsoid (spherical, so
                # all radii are the same)
                radius = np.random.randint(1, max_radius)

                ellipsoid_volume = ellipsoid(radius, radius, radius)
                zz_e, yy_e, xx_e = np.where(ellipsoid_volume)

                # Adjust coordinates based on the center
                zz_e += center_z - radius
                yy_e += center_y - radius
                xx_e += center_x - radius

                zz.extend(zz_e)
                yy.extend(yy_e)
                xx.extend(xx_e)

            zz = np.array(zz)
            yy = np.array(yy)
            xx = np.array(xx)

        valid_mask = (zz < z_size) & (yy < y_size) & (xx < x_size)

        return zz[valid_mask], yy[valid_mask], xx[valid_mask]

    def process(self, batch, request):
        data = batch.arrays[self.array].data

        # todo: 2d/3d
        assert len(data.shape) == 4, "Data must be 4-dimensional."

        # Choose a random number up to num_channels
        channels_to_zero_out = np.random.randint(1, self.num_channels + 1)

        # Randomly select the channels
        channels_to_zero = np.random.choice(
            data.shape[0], channels_to_zero_out, replace=False
        )

        # todo: parameterize
        use_shape = random.choice([True, False])  # 50/50 chance

        # todo: paramaterize
        set_values = [random.uniform(-0.03, 0.03) for i in range(100)]
        set_value = random.choice(set_values)

        if use_shape:
            _, z, y, x = data.shape

            # todo: parameterize?
            shape_type = np.random.choice(["ellipsoid", "points", "rectangle"])

            zz, yy, xx = self.draw_random_shape(z, y, x, shape_type=shape_type)

            # Ensure that the coordinates are within bounds
            valid_mask = (
                (zz >= 0) & (zz < z) & (yy >= 0) & (yy < y) & (xx >= 0) & (xx < x)
            )

            zz, yy, xx = zz[valid_mask], yy[valid_mask], xx[valid_mask]

            # Zero out selected channels in shape
            for channel in channels_to_zero:
                value = data[channel, zz, yy, xx]

                data[channel, zz, yy, xx] = value + set_value

        else:
            # whole selected channels
            data[channels_to_zero] = set_value

        batch[self.array].data = data
