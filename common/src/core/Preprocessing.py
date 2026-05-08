"""
Preprocessing
=============
Functions for preparing 3D microscopy image stacks prior to analysis,
including isotropic resampling along the Z axis and slice-wise bilateral
filtering.
"""

import numpy as np
import cv2


def matching_z_resolution(
    img_stack: np.ndarray,
    x_y_resolution: float,
    z_resolution: float,
) -> np.ndarray:
    """
    Resample a 3D stack so that voxel spacing along Z matches the in-plane
    resolution, using linear interpolation between consecutive slices.

    Parameters
    ----------
    img_stack : np.ndarray
        3D array of shape ``(Z, Y, X)``.
    x_y_resolution : float
        In-plane pixel size (assumed equal in X and Y).
    z_resolution : float
        Original slice spacing along Z.

    Returns
    -------
    np.ndarray
        Resampled stack with isotropic voxel spacing, same dtype as input.
    """
    current_depth, current_height, current_width = img_stack.shape

    scaling_factor = round(z_resolution / x_y_resolution)

    # Number of slices after inserting interpolated frames
    new_depth = (current_depth - 1) * scaling_factor + 1

    resized = np.zeros(
        (new_depth, current_height, current_width), dtype=img_stack.dtype
    )

    resized[0] = img_stack[0]

    slice_idx = 1
    for k in range(current_depth - 1):
        current_slice = img_stack[k]
        next_slice = img_stack[k + 1]

        # Interpolated slices between current and next (exclusive of endpoints)
        for t in np.linspace(0, 1, scaling_factor + 1)[1:-1]:
            resized[slice_idx] = (1 - t) * current_slice + t * next_slice
            slice_idx += 1

        # Original next slice
        resized[slice_idx] = next_slice
        slice_idx += 1

    resized[-1] = img_stack[-1]

    return resized


def bilateral_filter_3d(
    image: np.ndarray,
    d: int = 7,
    sigma_color: float = 30.0,
    sigma_space: float = 5.0,
) -> np.ndarray:
    """
    Apply a bilateral filter to each 2D slice of a 3D stack independently.

    Parameters
    ----------
    image : np.ndarray
        3D array of shape ``(Z, Y, X)`` with intensity values in [0, 255].
    d : int
        Diameter of the pixel neighbourhood used by the filter.
    sigma_color : float
        Filter sigma in the intensity/colour domain.  Larger values allow
        more dissimilar intensities to be mixed.
    sigma_space : float
        Filter sigma in the spatial domain.  Larger values increase the
        effective neighbourhood size.

    Returns
    -------
    np.ndarray
        Filtered stack (float32), same shape as input.
    """
    image_f32 = image.astype(np.float32)
    filtered = np.zeros_like(image_f32)

    for i in range(image_f32.shape[0]):
        filtered[i] = cv2.bilateralFilter(
            image_f32[i], d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space
        )

    return filtered