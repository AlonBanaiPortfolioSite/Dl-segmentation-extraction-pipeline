"""
Thresholding
=============
Hysteresis thresholding for 2D and 3D binary segmentation.  Voxels above
``high_thresh`` are foreground, voxels below ``low_thresh`` are background,
and voxels in between are assigned based on adjacency to confirmed foreground.
"""

import numpy as np


def hysteresis_threshold_3d(
    image: np.ndarray,
    low_thresh: float,
    high_thresh: float,
) -> np.ndarray:
    """
    Apply hysteresis thresholding to a 3D volume using 6-connectivity.

    Parameters
    ----------
    image : np.ndarray
        3D array of shape ``(Z, Y, X)``.
    low_thresh : float
        Values at or below this are background.
    high_thresh : float
        Values at or above this are foreground.

    Returns
    -------
    np.ndarray
        Binary mask (0/1, ``int``) of the same shape as *image*.

    Notes
    -----
    Mid-range voxels (``low_thresh < value < high_thresh``) are promoted to
    foreground only if at least one 6-connected neighbour is already
    foreground.  This is a single-pass heuristic — iterative propagation is
    not performed.
    """
    foreground = image >= high_thresh
    mid_mask = (image > low_thresh) & (image < high_thresh)
    mid_indices = np.argwhere(mid_mask)

    neighbours = [
        (-1, 0, 0), (1, 0, 0),
        (0, -1, 0), (0, 1, 0),
        (0, 0, -1), (0, 0, 1),
    ]

    dz, dy, dx = image.shape
    for z, y, x in mid_indices:
        for nz, ny, nx in neighbours:
            zn, yn, xn = z + nz, y + ny, x + nx
            if 0 <= zn < dz and 0 <= yn < dy and 0 <= xn < dx:
                if foreground[zn, yn, xn]:
                    foreground[z, y, x] = True
                    break

    return foreground.astype(int)


def hysteresis_threshold_2d_slicewise(
    image: np.ndarray,
    low_thresh: float,
    high_thresh: float,
) -> np.ndarray:
    """
    Apply hysteresis thresholding independently to each slice of a 3D stack,
    using 4-connectivity within each 2D slice.

    Parameters
    ----------
    image : np.ndarray
        3D array of shape ``(Z, Y, X)``.
    low_thresh : float
        Values at or below this are background.
    high_thresh : float
        Values at or above this are foreground.

    Returns
    -------
    np.ndarray
        Binary mask (0/1, ``int``) of the same shape as *image*.
    """
    mask = np.zeros_like(image, dtype=int)

    neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for i in range(image.shape[0]):
        slice_img = image[i]
        foreground = slice_img >= high_thresh
        mid_mask = (slice_img > low_thresh) & (slice_img < high_thresh)
        mid_indices = np.argwhere(mid_mask)

        h, w = slice_img.shape
        for y, x in mid_indices:
            for ny, nx in neighbours:
                yn, xn = y + ny, x + nx
                if 0 <= yn < h and 0 <= xn < w:
                    if foreground[yn, xn]:
                        foreground[y, x] = True
                        break

        mask[i] = foreground.astype(int)

    return mask