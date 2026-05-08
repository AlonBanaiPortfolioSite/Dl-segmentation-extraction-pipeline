"""
Morphological Operations
========================
3D morphological operations on binary masks, including dilation, erosion,
hole filling, connected-component analysis, and object filtering.
"""

import numpy as np
import cv2
from skimage import measure
from typing import Tuple, List


def dilation_3d(
    mask_stack: np.ndarray,
    kernel_size: int = 15,
) -> np.ndarray:
    """
    Apply 2D dilation slice-by-slice to a 3D binary mask.

    Parameters
    ----------
    mask_stack : np.ndarray
        Binary volume of shape ``(Z, Y, X)``.
    kernel_size : int
        Side length of the square structuring element.

    Returns
    -------
    np.ndarray
        Dilated mask, same shape as input.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated = np.zeros_like(mask_stack)
    for i in range(mask_stack.shape[0]):
        dilated[i] = cv2.dilate(mask_stack[i].astype(np.uint8), kernel, iterations=1)
    return dilated


def erosion_3d(
    mask_stack: np.ndarray,
    kernel_size: int = 15,
) -> np.ndarray:
    """
    Apply 2D erosion slice-by-slice to a 3D binary mask.

    Parameters
    ----------
    mask_stack : np.ndarray
        Binary volume of shape ``(Z, Y, X)``.
    kernel_size : int
        Side length of the square structuring element.

    Returns
    -------
    np.ndarray
        Eroded mask, same shape as input.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    eroded = np.zeros_like(mask_stack)
    for i in range(mask_stack.shape[0]):
        eroded[i] = cv2.erode(mask_stack[i].astype(np.uint8), kernel, iterations=1)
    return eroded


def find_connected_components_3d(
    mask_3d: np.ndarray,
    connectivity: int = 3,
) -> Tuple[np.ndarray, List[int]]:
    """
    Label connected components in a 3D binary mask.

    Parameters
    ----------
    mask_3d : np.ndarray
        Binary volume of shape ``(Z, Y, X)``.
    connectivity : int
        Connectivity for labelling (1 = face, 2 = face+edge, 3 = full 26-connected).

    Returns
    -------
    labeled_image : np.ndarray
        Integer array where each component has a unique label (1-based).
    component_sizes : list[int]
        Number of voxels in each component (ordered by label).
    """
    labeled_image, _ = measure.label(
        mask_3d, connectivity=connectivity, return_num=True
    )
    regions = measure.regionprops(labeled_image)
    component_sizes = [region.area for region in regions]
    return labeled_image, component_sizes


def remove_small_objects(
    mask: np.ndarray,
    min_size: int,
) -> np.ndarray:
    """
    Remove connected components smaller than *min_size* from a 3D binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary volume of shape ``(Z, Y, X)``.
    min_size : int
        Minimum number of voxels a component must contain to be retained.

    Returns
    -------
    np.ndarray
        Filtered binary mask (dtype ``bool``).
    """
    labeled_image, component_sizes = find_connected_components_3d(mask)
    filtered = np.zeros_like(mask, dtype=bool)
    for idx, size in enumerate(component_sizes):
        if size >= min_size:
            filtered |= (labeled_image == idx + 1)
    return filtered


def find_largest_object(
    mask: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """
    Extract the largest connected component from a 3D binary mask.

    Parameters
    ----------
    mask : np.ndarray
        Binary volume of shape ``(Z, Y, X)``.

    Returns
    -------
    main_object : np.ndarray
        Boolean mask containing only the largest component.
    object_size : int
        Number of voxels in that component.
    """
    labeled_image, component_sizes = find_connected_components_3d(mask)
    max_index = np.argmax(component_sizes)
    object_size = component_sizes[max_index]
    main_object = labeled_image == max_index + 1
    return main_object, object_size


# ---------------------------------------------------------------------------
# Hole filling
# ---------------------------------------------------------------------------

def imfill_holes(
    binary_mask: np.ndarray,
    mode: str = 'all_edges',
) -> np.ndarray:
    """
    Fill holes in a 2D binary mask using flood-fill from the background.

    Parameters
    ----------
    binary_mask : np.ndarray
        2D array with values 0/1 (or boolean).
    mode : {'all_edges', 'corner'}
        ``'all_edges'`` — every background pixel touching any image edge is
        treated as exterior; everything unreachable is filled.
        ``'corner'`` — only the four corner pixels seed the exterior flood
        fill.

    Returns
    -------
    np.ndarray
        Binary mask (0/1, ``uint8``) with holes filled.

    Raises
    ------
    ValueError
        If *mode* is not ``'all_edges'`` or ``'corner'``.
    """
    binary_mask = np.uint8(binary_mask > 0)
    inv_mask = cv2.bitwise_not(binary_mask * 255)
    h, w = inv_mask.shape
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)

    flood_filled = inv_mask.copy()

    if mode == 'all_edges':
        for row in range(h):
            if inv_mask[row, 0] == 255:
                cv2.floodFill(flood_filled, flood_mask, (0, row), 0)
            if inv_mask[row, w - 1] == 255:
                cv2.floodFill(flood_filled, flood_mask, (w - 1, row), 0)
        for col in range(w):
            if inv_mask[0, col] == 255:
                cv2.floodFill(flood_filled, flood_mask, (col, 0), 0)
            if inv_mask[h - 1, col] == 255:
                cv2.floodFill(flood_filled, flood_mask, (col, h - 1), 0)

    elif mode == 'corner':
        corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
        for row, col in corners:
            if inv_mask[row, col] == 255:
                cv2.floodFill(flood_filled, flood_mask, (col, row), 0)

    else:
        raise ValueError("mode must be 'all_edges' or 'corner'")

    filled = cv2.bitwise_or(binary_mask * 255, flood_filled)
    return filled // 255


def imfill_holes_3d(
    mask_stack: np.ndarray,
    mode: str = 'corner',
) -> np.ndarray:
    """
    Apply :func:`imfill_holes` to each slice of a 3D mask independently.

    Parameters
    ----------
    mask_stack : np.ndarray
        Binary volume of shape ``(Z, Y, X)``.
    mode : {'all_edges', 'corner'}
        Flood-fill seeding strategy (see :func:`imfill_holes`).

    Returns
    -------
    np.ndarray
        Hole-filled mask, same shape as input.
    """
    filled = np.zeros_like(mask_stack)
    for i in range(mask_stack.shape[0]):
        filled[i] = imfill_holes(mask_stack[i], mode)
    return filled