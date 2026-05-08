"""
Active Contour Refinement
=========================
Slice-by-slice active contour (snake) refinement of an initial segmentation
mask on a denoised 3D image stack.

.. note::
    This module was used for mid-pipeline contour refinement during
    retraining iterations.  It is **not part of the final segmentation
    pipeline** but is retained for reproducibility.
"""

import os
import time
from typing import Tuple

import imageio.v3 as iio
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import gaussian
from skimage.segmentation import active_contour

from src.core.io_utils import save_3d_stack_as_tiff
from src.core.preprocessing import bilateral_filter_3d

from .config import ExperimentConfig
from .utils import (
    calculate_slice_statistics,
    create_experiment_log,
    draw_contour_filled,
    draw_contour_perimeter,
    extract_contour_coordinates,
    save_summary_statistics,
)


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------

def load_image_based_on_format(
    filepath: str,
    file_format: str,
    axis_order: str,
) -> np.ndarray:
    """
    Load an image file and ensure ``(Z, Y, X)`` axis order.

    Parameters
    ----------
    filepath : str
        Path to the image.
    file_format : {'tiff', 'nifti'}
        File format.
    axis_order : {'zyx', 'xyz'}
        Axis order of the stored data.

    Returns
    -------
    np.ndarray
        Image in ``(Z, Y, X)`` layout.
    """
    print(f"Loading {file_format} file: {filepath}")

    if file_format == "tiff":
        image = np.asarray(iio.imread(filepath))
    elif file_format == "nifti":
        import nibabel as nib
        image = nib.load(filepath).get_fdata()
    else:
        raise ValueError(f"Unknown file format: {file_format}")

    if axis_order == "xyz":
        print(f"  Transposing from (X, Y, Z) {image.shape}", end="")
        image = np.transpose(image, (2, 1, 0))
        print(f" -> (Z, Y, X) {image.shape}")
    else:
        print(f"  Shape (Z, Y, X): {image.shape}")

    return image


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def apply_filter_3d(
    image_stack: np.ndarray,
    config: ExperimentConfig,
) -> np.ndarray:
    """
    Apply the configured denoising filter to a 3D image stack.

    Parameters
    ----------
    image_stack : np.ndarray
        3D array ``(Z, Y, X)``.
    config : ExperimentConfig
        Experiment configuration (filter parameters are read from here).

    Returns
    -------
    np.ndarray
        Filtered stack, normalised to [0, 1] for bilateral filtering or
        as returned by ``skimage.filters.gaussian``.
    """
    ftype = config.filter.filter_type
    print(f"Applying {ftype} filter ...")

    if ftype == "gaussian":
        filtered = gaussian(image_stack, sigma=config.filter.sigma, preserve_range=False)

    elif ftype == "bilateral":
        filtered = bilateral_filter_3d(
            image_stack,
            d=config.filter.d,
            sigma_color=config.filter.sigma_color,
            sigma_space=config.filter.sigma_space,
        )
        fmin, fmax = filtered.min(), filtered.max()
        if fmax > fmin:
            filtered = (filtered - fmin) / (fmax - fmin)
    else:
        raise ValueError(f"Unknown filter type: {ftype}")

    print(f"  Output range: [{filtered.min():.3f}, {filtered.max():.3f}]")
    return filtered


# ---------------------------------------------------------------------------
# Contour refinement
# ---------------------------------------------------------------------------

_MAX_POINT_DISPLACEMENT = 30  # pixels — outlier threshold


def refine_contour_3d(
    image_stack: np.ndarray,
    init_contour_stack: np.ndarray,
    denoised_stack: np.ndarray,
    config: ExperimentConfig,
    verbose: bool = True,
) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Refine an initial contour mask using active contours, slice by slice.

    Parameters
    ----------
    image_stack : np.ndarray
        Original 3D image ``(Z, Y, X)``.
    init_contour_stack : np.ndarray
        Binary mask of the initial contour ``(Z, Y, X)``.
    denoised_stack : np.ndarray
        Denoised image used as the snake energy landscape.
    config : ExperimentConfig
        Experiment configuration.
    verbose : bool
        Print per-slice progress.

    Returns
    -------
    init_contour_perimeter : np.ndarray
        Perimeter rendering of the initial contours.
    final_contour_filled : np.ndarray
        Filled mask of the refined contours.
    slice_stats : list[dict]
        Per-slice displacement statistics.
    """
    print("Starting active contour refinement ...")

    init_contour_perimeter = np.zeros_like(init_contour_stack, dtype=np.uint8)
    final_contour_filled = np.zeros_like(init_contour_stack, dtype=np.uint8)
    slice_stats: list = []

    num_slices = image_stack.shape[0]

    for z in range(num_slices):
        if verbose:
            print(f"\n{'=' * 50}")
            print(f"Slice {z + 1}/{num_slices}")

        init_coords = extract_contour_coordinates(init_contour_stack[z], threshold=0.5)

        if init_coords is None:
            print(f"  Warning: no contour in slice {z}")
            slice_stats.append({
                "slice": z, "warning": True,
                "mean_distance": 0, "max_distance": 0,
                "min_distance": 0, "std_distance": 0, "num_points": 0,
            })
            continue

        if verbose:
            print(f"  Init contour: {len(init_coords)} points")

        init_contour_perimeter[z] = draw_contour_perimeter(
            image_stack.shape[1:], init_coords,
        )

        denoised_slice = denoised_stack[z]

        if verbose:
            print(f"  Image range: [{denoised_slice.min():.3f}, {denoised_slice.max():.3f}]")

        try:
            snake = active_contour(
                denoised_slice,
                init_coords,
                alpha=config.active_contour.alpha,
                beta=config.active_contour.beta,
                gamma=config.active_contour.gamma,
            )

            distances = np.linalg.norm(snake - init_coords, axis=1)
            outliers = distances > _MAX_POINT_DISPLACEMENT

            if np.any(outliers):
                keep = ~outliers
                snake_filtered = snake[keep]
                init_filtered = init_coords[keep]
                if verbose:
                    print(f"  Removed {outliers.sum()} outlier points, "
                          f"kept {keep.sum()}")
                stats = calculate_slice_statistics(init_filtered, snake_filtered)
                stats["outliers_removed"] = int(outliers.sum())
                final_contour_filled[z] = draw_contour_filled(
                    image_stack.shape[1:], snake_filtered,
                )
            else:
                stats = calculate_slice_statistics(init_coords, snake)
                stats["outliers_removed"] = 0
                final_contour_filled[z] = draw_contour_filled(
                    image_stack.shape[1:], snake,
                )

            stats["slice"] = z
            stats["warning"] = False
            slice_stats.append(stats)

            if verbose:
                print(f"  Mean displacement: {stats['mean_distance']:.2f} px, "
                      f"max: {stats['max_distance']:.2f} px")

        except Exception as exc:
            print(f"  Error in slice {z}: {exc}")
            slice_stats.append({
                "slice": z, "warning": True,
                "mean_distance": 0, "max_distance": 0,
                "min_distance": 0, "std_distance": 0, "num_points": 0,
            })

    ok = sum(1 for s in slice_stats if not s["warning"])
    print(f"\nRefinement complete — {ok}/{num_slices} slices processed.")

    return init_contour_perimeter, final_contour_filled, slice_stats


# ---------------------------------------------------------------------------
# Full experiment runner
# ---------------------------------------------------------------------------

def run_active_contour_experiment(
    image_stack: np.ndarray,
    init_contour_stack: np.ndarray,
    config: ExperimentConfig,
    verbose: bool = True,
) -> dict:
    """
    End-to-end active contour refinement experiment.

    Steps: filter → refine contours → save outputs → write log.

    Parameters
    ----------
    image_stack : np.ndarray
        Original 3D image ``(Z, Y, X)``.
    init_contour_stack : np.ndarray
        Binary mask of the initial contour.
    config : ExperimentConfig
        Experiment configuration.
    verbose : bool
        Print progress.

    Returns
    -------
    dict
        Keys: ``config``, ``denoised_stack``, ``init_contour_perimeter``,
        ``final_contour_filled``, ``slice_stats``, and paths to saved files.
    """
    start_time = time.time()

    print("=" * 60)
    print("ACTIVE CONTOUR REFINEMENT EXPERIMENT")
    print("=" * 60)
    print(f"Experiment : {config.experiment_name}")
    print(f"Output     : {config.output_folder}")
    print(f"Stack shape: {image_stack.shape}")

    os.makedirs(config.output_folder, exist_ok=True)
    config.save_to_yaml()

    # 1. Filter
    denoised_stack = apply_filter_3d(image_stack, config)

    # 2. Refine
    init_perimeter, final_filled, slice_stats = refine_contour_3d(
        image_stack, init_contour_stack, denoised_stack, config, verbose,
    )

    # 3. Save
    results: dict = {
        "config": config,
        "denoised_stack": denoised_stack,
        "init_contour_perimeter": init_perimeter,
        "final_contour_filled": final_filled,
        "slice_stats": slice_stats,
    }

    if config.output.save_denoised_image:
        orig_min, orig_max = image_stack.min(), image_stack.max()
        rescaled = (denoised_stack * (orig_max - orig_min) + orig_min).astype(np.float32)
        save_3d_stack_as_tiff(rescaled, config.output_folder, "denoised_image.tiff")

    if config.output.save_init_contour:
        save_3d_stack_as_tiff(init_perimeter, config.output_folder, "init_contour_perimeter.tiff")

    if config.output.save_final_contour:
        binary = (final_filled > 0).astype(np.uint8)
        save_3d_stack_as_tiff(binary, config.output_folder, "final_contour_filled.tiff")

    if config.output.save_summary_stats:
        results["stats_path"] = save_summary_statistics(slice_stats, config.output_folder)

    # 4. Log
    processing_time = time.time() - start_time
    results["log_path"] = create_experiment_log(
        config, config.output_folder, processing_time, slice_stats,
    )

    print(f"\nExperiment complete — {processing_time:.2f} s")
    print(f"Results in: {config.output_folder}")

    return results