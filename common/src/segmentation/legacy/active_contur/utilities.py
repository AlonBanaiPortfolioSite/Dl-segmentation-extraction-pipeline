"""
Active Contour Utilities
========================
Helper functions for contour extraction, drawing, and statistics used by the
active contour refinement pipeline.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from skimage import measure
from skimage.draw import polygon, polygon_perimeter


# ---------------------------------------------------------------------------
# Contour extraction & drawing
# ---------------------------------------------------------------------------

def extract_contour_coordinates(
    binary_image: np.ndarray,
    threshold: float = 0.5,
) -> Optional[np.ndarray]:
    """
    Extract the largest contour from a 2D binary image.

    Parameters
    ----------
    binary_image : np.ndarray
        2D binary image.
    threshold : float
        Iso-value for ``skimage.measure.find_contours``.

    Returns
    -------
    np.ndarray or None
        (N, 2) array of contour coordinates, or ``None`` if no contour is
        found.
    """
    contours = measure.find_contours(binary_image, threshold)
    if len(contours) == 0:
        return None
    return max(contours, key=len)


def draw_contour_filled(
    image_shape: Tuple[int, int],
    contour_coords: np.ndarray,
) -> np.ndarray:
    """
    Rasterise a filled polygon from contour coordinates.

    Parameters
    ----------
    image_shape : tuple[int, int]
        (height, width) of the output image.
    contour_coords : np.ndarray
        (N, 2) contour coordinates (row, col).

    Returns
    -------
    np.ndarray
        ``uint8`` image with the polygon interior set to 255.
    """
    image = np.zeros(image_shape, dtype=np.uint8)
    coords = contour_coords.astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, image_shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, image_shape[1] - 1)
    rr, cc = polygon(coords[:, 0], coords[:, 1], image_shape)
    image[rr, cc] = 255
    return image


def draw_contour_perimeter(
    image_shape: Tuple[int, int],
    contour_coords: np.ndarray,
) -> np.ndarray:
    """
    Rasterise the outline of a contour.

    Parameters
    ----------
    image_shape : tuple[int, int]
        (height, width) of the output image.
    contour_coords : np.ndarray
        (N, 2) contour coordinates (row, col).

    Returns
    -------
    np.ndarray
        ``uint8`` image with the perimeter pixels set to 255.
    """
    image = np.zeros(image_shape, dtype=np.uint8)
    coords = contour_coords.astype(int)
    coords[:, 0] = np.clip(coords[:, 0], 0, image_shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, image_shape[1] - 1)
    rr, cc = polygon_perimeter(coords[:, 0], coords[:, 1], image_shape)
    image[rr, cc] = 255
    return image


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def calculate_slice_statistics(
    init_contour: np.ndarray,
    final_contour: np.ndarray,
) -> Dict[str, float]:
    """
    Compute displacement statistics between two contours of equal length.

    Parameters
    ----------
    init_contour, final_contour : np.ndarray
        (N, 2) coordinate arrays (must have the same *N*).

    Returns
    -------
    dict
        Keys: ``mean_distance``, ``max_distance``, ``min_distance``,
        ``std_distance``, ``num_points``.
    """
    distances = np.linalg.norm(final_contour - init_contour, axis=1)
    return {
        "mean_distance": float(distances.mean()),
        "max_distance": float(distances.max()),
        "min_distance": float(distances.min()),
        "std_distance": float(distances.std()),
        "num_points": len(distances),
    }


def save_summary_statistics(
    slice_stats: List[dict],
    output_folder: str,
    filename: str = "summary_stats.csv",
) -> str:
    """
    Write per-slice statistics to a CSV file.

    Parameters
    ----------
    slice_stats : list[dict]
        One dictionary per slice (as returned by
        :func:`calculate_slice_statistics` with extra fields).
    output_folder : str
        Destination directory.
    filename : str
        CSV filename.

    Returns
    -------
    str
        Path to the saved CSV.
    """
    os.makedirs(output_folder, exist_ok=True)

    df = pd.DataFrame(slice_stats)
    filepath = os.path.join(output_folder, filename)
    df.to_csv(filepath, index=False)

    warnings = int(df["warning"].sum()) if "warning" in df.columns else 0
    print(f"Statistics saved to: {filepath}")
    print(f"  Mean displacement: {df['mean_distance'].mean():.2f} px, "
          f"slices: {len(df)}, warnings: {warnings}")

    return filepath


# ---------------------------------------------------------------------------
# Experiment log
# ---------------------------------------------------------------------------

def create_experiment_log(
    config,
    output_folder: str,
    processing_time: float,
    slice_stats: List[dict],
) -> str:
    """
    Write a human-readable text log summarising an experiment run.

    Parameters
    ----------
    config : ExperimentConfig
        The experiment configuration object.
    output_folder : str
        Destination directory.
    processing_time : float
        Wall-clock time in seconds.
    slice_stats : list[dict]
        Per-slice statistics.

    Returns
    -------
    str
        Path to the log file.
    """
    log_path = os.path.join(output_folder, "experiment_log.txt")
    df = pd.DataFrame(slice_stats)
    warnings = int(df["warning"].sum()) if "warning" in df.columns else 0

    with open(log_path, "w") as fh:
        fh.write("=" * 60 + "\n")
        fh.write("ACTIVE CONTOUR REFINEMENT EXPERIMENT LOG\n")
        fh.write("=" * 60 + "\n\n")

        fh.write(f"Experiment Name : {config.experiment_name}\n")
        fh.write(f"Timestamp       : {config.timestamp}\n")
        fh.write(f"Processing Time : {processing_time:.2f} s\n\n")

        fh.write("FILTER CONFIGURATION:\n")
        fh.write(f"  Type: {config.filter.filter_type}\n")
        if config.filter.filter_type == "gaussian":
            fh.write(f"  Sigma: {config.filter.sigma}\n")
        else:
            fh.write(f"  d: {config.filter.d}\n")
            fh.write(f"  sigma_color: {config.filter.sigma_color}\n")
            fh.write(f"  sigma_space: {config.filter.sigma_space}\n")

        fh.write("\nACTIVE CONTOUR CONFIGURATION:\n")
        fh.write(f"  Alpha: {config.active_contour.alpha}\n")
        fh.write(f"  Beta : {config.active_contour.beta}\n")
        fh.write(f"  Gamma: {config.active_contour.gamma}\n")

        fh.write("\nPROCESSING RESULTS:\n")
        fh.write(f"  Total slices              : {len(slice_stats)}\n")
        fh.write(f"  Avg mean distance moved   : {df['mean_distance'].mean():.2f} px\n")
        fh.write(f"  Avg max distance moved    : {df['max_distance'].mean():.2f} px\n")
        fh.write(f"  Slices with warnings      : {warnings}\n")

        fh.write("\n" + "=" * 60 + "\n")

    print(f"Experiment log saved to: {log_path}")
    return log_path