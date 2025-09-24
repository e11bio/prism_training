from funlib.persistence import open_ds
from pathlib import Path
from scipy.ndimage import binary_erosion, distance_transform_edt
import numpy as np
import zarr


def expand_labels(labels, background=0, expansion_factor=1):
    expanded_labels = np.zeros_like(labels)

    z_slices = labels.shape[0]

    for z in range(z_slices):
        z_slice = labels[z]

        distances, indices = distance_transform_edt(
            z_slice == background, return_indices=True
        )

        dilate_mask = distances <= expansion_factor
        masked_indices = [
            dimension_indices[dilate_mask] for dimension_indices in indices
        ]
        nearest_labels = z_slice[tuple(masked_indices)]

        expanded_labels[z][dilate_mask] = nearest_labels

    return expanded_labels


def erode_labels(labels, background_label=0, steps=1):
    labels = np.copy(labels)

    foreground = np.zeros_like(labels, dtype=bool)

    for label in np.unique(labels):
        if label == background_label:
            continue
        label_mask = labels == label

        eroded_label_mask = binary_erosion(label_mask, iterations=steps, border_value=1)
        foreground = np.logical_or(eroded_label_mask, foreground)

    labels[np.logical_not(foreground)] = background_label

    return labels


def create_avg_intensity_barcodes(raw, labels):
    C = raw.shape[0]
    lbl = labels.astype(np.int64, copy=False)
    L = int(lbl.max())
    flat_lbl = lbl.ravel()

    counts = np.bincount(flat_lbl, minlength=L + 1).astype(np.float64)

    out = np.empty_like(raw, dtype=raw.dtype)
    for c in range(C):
        print(c)
        sums = np.bincount(flat_lbl, weights=raw[c].ravel(), minlength=L + 1)
        means = np.divide(
            sums, counts, out=np.zeros_like(sums, dtype=np.float64), where=counts > 0
        )
        means[0] = 0.0
        out[c] = means[lbl]

    return out


if __name__ == "__main__":
    store = Path("instance/example_data.zarr")

    raw = open_ds(store / "raw")
    labels = open_ds(store / "labels")

    voxel_size = raw.voxel_size

    roi = labels.roi

    raw_data = raw[roi]
    labels_data = labels[roi]

    # create unlabelled mask here instead of after erosion, helps to decrease
    # blurring boundaries between barcodes
    unlabelled = (labels_data > 0).astype(np.uint8)

    # expand labels
    print("expanding label boundaries...")
    labels_data = expand_labels(labels_data, expansion_factor=1)

    # then erode
    print("eroding label boundaries...")
    for z in range(labels.shape[0]):
        labels_data[z] = erode_labels(labels_data[z], steps=1)

    # normalize first
    raw_data = (raw_data / 255).astype(np.float32)

    # compute avg barcodes
    print("computing avg barcodes...")
    avg_barcodes = create_avg_intensity_barcodes(raw_data, labels_data)

    # scale back
    avg_barcodes = (avg_barcodes * 255).astype(np.uint8)

    container = zarr.open(store, "a")

    container["unlabelled"] = unlabelled
    container["unlabelled"].attrs["offset"] = roi.get_begin()
    container["unlabelled"].attrs["resolution"] = voxel_size

    container["avg_barcodes"] = avg_barcodes
    container["avg_barcodes"].attrs["offset"] = roi.get_begin()
    container["avg_barcodes"].attrs["resolution"] = voxel_size
