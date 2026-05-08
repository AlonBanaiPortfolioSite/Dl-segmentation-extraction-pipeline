"""
I/O Utilities
=============
Functions for loading, saving, and extracting metadata from 3D microscopy
image stacks (TIFF / LSM formats).
"""

import os
import time
import numpy as np
import tifffile
import imageio.v3 as iio
from typing import Tuple


def save_3d_stack_as_tiff(
    image_stack: np.ndarray,
    folder_path: str,
    filename: str,
    preserve_dtype: bool = True,
    uint8: bool = False,
    max_retries: int = 3,
) -> None:
    """
    Save a 3D image stack as a TIFF file.

    Parameters
    ----------
    image_stack : np.ndarray
        3D array (Z, Y, X).
    folder_path : str
        Destination directory (created if it does not exist).
    filename : str
        Output filename including the ``.tiff`` extension.
    preserve_dtype : bool
        If True, save with the original data type (recommended).
        When False, the ``uint8`` flag controls conversion.
    uint8 : bool
        If True **and** ``preserve_dtype`` is False, clip values to
        [0, 255] and convert to ``uint8``.  Ignored when
        ``preserve_dtype`` is True.
    max_retries : int
        Number of retry attempts for filesystem operations (useful when
        the target directory is on a synced drive).

    Raises
    ------
    ValueError
        If ``image_stack`` is not three-dimensional.
    OSError / PermissionError
        If all write attempts fail.
    """
    image_stack = np.asarray(image_stack)
    if image_stack.ndim != 3:
        raise ValueError("Input image_stack must be a 3D numpy array.")

    # Data-type handling
    if preserve_dtype:
        output_stack = image_stack
    elif uint8:
        output_stack = np.clip(image_stack, 0, 255).astype(np.uint8)
    else:
        output_stack = image_stack

    file_path = os.path.join(folder_path, filename)

    if len(file_path) > 260:
        print(
            f"Warning: path length ({len(file_path)}) exceeds the Windows "
            "260-character limit. Consider shortening the path."
        )

    for attempt in range(max_retries):
        try:
            os.makedirs(folder_path, exist_ok=True)
            if not os.path.exists(folder_path):
                raise OSError(f"Failed to create directory: {folder_path}")

            tifffile.imwrite(file_path, output_stack)

            print(f"Saved: {file_path}")
            print(f"  dtype={output_stack.dtype}  shape={output_stack.shape}  "
                  f"range=[{output_stack.min():.2f}, {output_stack.max():.2f}]")
            return

        except (OSError, PermissionError, FileNotFoundError) as exc:
            print(f"Attempt {attempt + 1}/{max_retries} failed: {exc}")
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                print(f"Debug info — folder: {folder_path}, "
                      f"exists: {os.path.exists(folder_path)}, "
                      f"path length: {len(file_path)}")
                raise


def load_image_stack(path: str, channel_num: int = 0) -> np.ndarray:
    """
    Load a multi-channel 3D image stack and return a single channel.

    Assumes the file has shape ``(Z, C, Y, X)`` where *C* is the channel
    axis.

    Parameters
    ----------
    path : str
        Path to the image file (TIFF, LSM, etc.).
    channel_num : int
        Zero-based channel index to extract.

    Returns
    -------
    np.ndarray
        3D array (Z, Y, X) for the requested channel.
    """
    image_3D = np.asarray(iio.imread(path))
    return image_3D[:, channel_num, :, :]


def get_resolution_from_tiff(image_path: str) -> Tuple[float, float, float]:
    """
    Extract spatial resolution from a TIFF file.

    X/Y resolution is read from the TIFF ``XResolution`` / ``YResolution``
    tags (assumed to be in pixels-per-micron).  Z spacing is read from
    ``imageio`` metadata (``spacing`` key, defaults to 1.0 if absent).

    Parameters
    ----------
    image_path : str
        Path to the TIFF file.

    Returns
    -------
    x_microns : float
        Pixel size along X (microns).
    y_microns : float
        Pixel size along Y (microns).
    z_microns : float
        Slice spacing along Z (microns).
    """
    with tifffile.TiffFile(image_path) as tif:
        page = tif.pages[0]

        x_num, x_den = page.tags['XResolution'].value
        y_num, y_den = page.tags['YResolution'].value

        x_microns = x_den / x_num  # invert pixels/micron → microns/pixel
        y_microns = y_den / y_num

    meta = iio.immeta(image_path)
    z_microns = meta.get('spacing', 1.0)

    return x_microns, y_microns, z_microns