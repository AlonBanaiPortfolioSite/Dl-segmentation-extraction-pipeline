"""
Shape Analysis Module
=====================
3D shape feature extraction from binary segmentation masks.
Computes volume, surface area, and sphericity for connected components
using marching cubes mesh reconstruction on isotropically resampled masks.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from skimage.measure import marching_cubes
from src.morphology.operations import find_connected_components_3D
from src.core.preprocessing import matching_z_resolution


def shape_analysis(
    mask_3D: np.ndarray,
    x_y_resolution: float,
    z_resolution: float,
    mode: str = 'full',
    verbose: bool = False
) -> Tuple[int, List[float], List[float], List[float], List[float]]:
    """
    Analyze 3D shape properties of connected components in a binary mask.

    Each component is isotropically resampled via ``matching_z_resolution``
    before marching-cubes surface reconstruction.  Volumes are computed both
    from voxel counts and from the reconstructed mesh (signed-volume method).

    Parameters
    ----------
    mask_3D : np.ndarray
        Binary 3D mask (non-zero = foreground).
    x_y_resolution : float
        In-plane voxel size (mm).
    z_resolution : float
        Slice spacing (mm).
    mode : {'full', 'fast'}
        ``'full'``  — compute surface area / sphericity for every object.
        ``'fast'``  — compute only for the 3 largest objects (the rest are 0).
    verbose : bool
        Print per-object diagnostics.

    Returns
    -------
    num_of_obj : int
        Total number of connected components.
    volume_lst : list[float]
        Voxel-based volumes (mm³) for all objects.
    volume_mc_lst : list[float]
        Marching-cubes mesh volumes (mm³).
    surface_area_mc_lst : list[float]
        Marching-cubes surface areas (mm²).
    sphericity_mc_lst : list[float]
        Sphericity values derived from mesh volume and surface area.
    """
    labeled_image, component_sizes = find_connected_components_3D(mask_3D)
    num_of_obj = len(component_sizes)

    # Voxel-based volumes (fast for all objects)
    volume_lst = [
        size * x_y_resolution ** 2 * z_resolution for size in component_sizes
    ]

    if verbose:
        sorted_indices = np.argsort(volume_lst)[::-1]
        print(f"\nTotal objects found: {num_of_obj}")
        print(f"\nTop 10 largest objects by volume:")
        for rank, idx in enumerate(sorted_indices[:10], 1):
            print(f"  {rank}. Object {idx + 1}: Volume = {volume_lst[idx]:.2f} mm³")

    # Determine which objects get full mesh analysis
    if mode == 'fast':
        sorted_indices = np.argsort(volume_lst)[::-1]
        objects_to_analyze = sorted_indices[:3].tolist()
        if verbose:
            print("\nFast mode: analyzing surface area for 3 largest objects only")
    elif mode == 'full':
        objects_to_analyze = list(range(num_of_obj))
        if verbose:
            print(f"\nFull mode: analyzing surface area for all {num_of_obj} objects")
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'full' or 'fast'")

    # Pre-allocate with zeros (uncalculated objects stay at 0)
    surface_area_mc_lst = [0.0] * num_of_obj
    sphericity_mc_lst = [0.0] * num_of_obj
    volume_mc_lst = [0.0] * num_of_obj

    for i in objects_to_analyze:
        # Extract single component and resample to isotropic resolution
        component_mask = (labeled_image == i + 1).astype(np.uint8)
        component_mask = matching_z_resolution(
            img_stack=component_mask,
            x_y_resolution=x_y_resolution,
            z_resolution=z_resolution,
        )

        if component_mask.max() == 0 or component_mask.sum() < 3:
            continue

        # Marching cubes surface reconstruction
        verts, faces, _, _ = marching_cubes(component_mask, level=0.5)
        tris = verts[faces]

        # Surface area (vectorised cross-product method)
        edge1 = tris[:, 1, :] - tris[:, 0, :]
        edge2 = tris[:, 2, :] - tris[:, 0, :]
        cross_products = np.cross(edge1, edge2)
        triangle_areas = 0.5 * np.linalg.norm(cross_products, axis=1)
        surface_area = triangle_areas.sum() * x_y_resolution ** 2
        surface_area_mc_lst[i] = surface_area

        # Mesh volume (signed-volume / divergence-theorem method)
        v0 = tris[:, 0, :]
        signed_volumes = np.sum(v0 * cross_products, axis=1) / 6.0
        volume_mc = np.abs(signed_volumes.sum()) * x_y_resolution ** 3
        volume_mc_lst[i] = volume_mc

        # Sphericity
        if surface_area > 0:
            sphericity = (
                (np.pi ** (1 / 3) * (6 * volume_mc) ** (2 / 3)) / surface_area
            )
            sphericity_mc_lst[i] = sphericity

        if verbose and mode == 'fast':
            print(
                f"  Object {i + 1}: SA = {surface_area:.2f} mm², "
                f"Sphericity = {sphericity_mc_lst[i]:.3f}"
            )

    return (
        num_of_obj,
        volume_lst,
        volume_mc_lst,
        surface_area_mc_lst,
        sphericity_mc_lst,
    )


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def extract_shape_features(
    vols: List[float],
    vols_mc: List[float],
    areas_mc: List[float],
    sphs_mc: List[float],
    mode: str = 'tumor',
) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame from per-object shape measurements.

    Parameters
    ----------
    vols : list[float]
        Voxel-based volumes for all objects.
    vols_mc : list[float]
        Marching-cubes mesh volumes for all objects.
    areas_mc : list[float]
        Surface areas (may contain zeros for uncalculated objects).
    sphs_mc : list[float]
        Sphericity values (may contain zeros for uncalculated objects).
    mode : {'tumor', 'invading_cells'}
        ``'tumor'``          — main / second / rest-average feature hierarchy.
        ``'invading_cells'`` — top-3 largest calculated objects.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with extracted features.
    """
    if mode == 'tumor':
        return _extract_tumor_features(vols, vols_mc, areas_mc, sphs_mc)
    elif mode == 'invading_cells':
        return _extract_invading_cells_features(vols, vols_mc, areas_mc, sphs_mc)
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'tumor' or 'invading_cells'")


def _extract_invading_cells_features(
    vols: List[float],
    vols_mc: List[float],
    areas_mc: List[float],
    sphs_mc: List[float],
) -> pd.DataFrame:
    """Extract features for invading-cells analysis (top-3 largest objects)."""
    calculated_mask = np.array(areas_mc) > 0
    calculated_vols = np.array(vols)[calculated_mask]
    calculated_vols_mc = np.array(vols_mc)[calculated_mask]
    calculated_areas = np.array(areas_mc)[calculated_mask]
    calculated_sphs = np.array(sphs_mc)[calculated_mask]

    sorted_indices = np.argsort(calculated_vols)[::-1]

    features: Dict[str, float] = {
        'total_volume': sum(vols),
        'total_volume_mc': sum(vols_mc),
        'num_objects': len(vols),
        'num_calculated_objects': int(calculated_mask.sum()),
    }

    for rank in range(min(3, len(calculated_vols))):
        idx = sorted_indices[rank]
        features[f'largest_obj_{rank + 1}_volume'] = calculated_vols[idx]
        features[f'largest_obj_{rank + 1}_volume_mc'] = calculated_vols_mc[idx]
        features[f'largest_obj_{rank + 1}_surface_area'] = calculated_areas[idx]
        features[f'largest_obj_{rank + 1}_sphericity'] = calculated_sphs[idx]

    for rank in range(len(calculated_vols), 3):
        features[f'largest_obj_{rank + 1}_volume'] = 0.0
        features[f'largest_obj_{rank + 1}_volume_mc'] = 0.0
        features[f'largest_obj_{rank + 1}_surface_area'] = 0.0
        features[f'largest_obj_{rank + 1}_sphericity'] = 0.0

    return pd.DataFrame([features])


def _extract_tumor_features(
    vols: List[float],
    vols_mc: List[float],
    areas_mc: List[float],
    sphs_mc: List[float],
) -> pd.DataFrame:
    """Extract tumour features with main / second / rest-average hierarchy."""
    df = pd.DataFrame({
        'Volume': vols,
        'Volume_MC': vols_mc,
        'Surface_Area_MC': areas_mc,
        'Sphericity_MC': sphs_mc,
    })

    # Keep only objects for which mesh analysis was performed
    df = df[df['Surface_Area_MC'] > 0]

    if df.empty:
        return pd.DataFrame([{
            'main_tumor_volume': None, 'main_tumor_volume_mc': None,
            'main_tumor_surface_area': None, 'main_tumor_sphericity': None,
            'second_tumor_volume': None, 'second_tumor_volume_mc': None,
            'second_tumor_surface_area': None, 'second_tumor_sphericity': None,
            'avg_rest_volume': None, 'avg_rest_volume_mc': None,
            'avg_rest_surface_area': None, 'avg_rest_sphericity': None,
            'num_of_tumors': 0,
            'total_volume': 0, 'total_volume_mc': 0, 'total_surface_area': 0,
            'median_volume': None, 'median_volume_mc': None,
            'median_surface_area': None,
        }])

    sorted_df = df.sort_values('Volume', ascending=False).reset_index(drop=True)

    features: Dict[str, Optional[float]] = {
        'num_of_tumors': len(sorted_df),
        'total_volume': sorted_df['Volume'].sum(),
        'total_volume_mc': sorted_df['Volume_MC'].sum(),
        'total_surface_area': sorted_df['Surface_Area_MC'].sum(),
        'median_volume': sorted_df['Volume'].median(),
        'median_volume_mc': sorted_df['Volume_MC'].median(),
        'median_surface_area': sorted_df['Surface_Area_MC'].median(),
    }

    # Main tumour (largest)
    features['main_tumor_volume'] = sorted_df.iloc[0]['Volume']
    features['main_tumor_volume_mc'] = sorted_df.iloc[0]['Volume_MC']
    features['main_tumor_surface_area'] = sorted_df.iloc[0]['Surface_Area_MC']
    features['main_tumor_sphericity'] = sorted_df.iloc[0]['Sphericity_MC']

    # Second largest
    if len(sorted_df) > 1:
        features['second_tumor_volume'] = sorted_df.iloc[1]['Volume']
        features['second_tumor_volume_mc'] = sorted_df.iloc[1]['Volume_MC']
        features['second_tumor_surface_area'] = sorted_df.iloc[1]['Surface_Area_MC']
        features['second_tumor_sphericity'] = sorted_df.iloc[1]['Sphericity_MC']
    else:
        features['second_tumor_volume'] = None
        features['second_tumor_volume_mc'] = None
        features['second_tumor_surface_area'] = None
        features['second_tumor_sphericity'] = None

    # Remaining tumours average
    if len(sorted_df) > 2:
        rest = sorted_df.iloc[2:]
        features['avg_rest_volume'] = rest['Volume'].mean()
        features['avg_rest_volume_mc'] = rest['Volume_MC'].mean()
        features['avg_rest_surface_area'] = rest['Surface_Area_MC'].mean()
        features['avg_rest_sphericity'] = rest['Sphericity_MC'].mean()
    else:
        features['avg_rest_volume'] = None
        features['avg_rest_volume_mc'] = None
        features['avg_rest_surface_area'] = None
        features['avg_rest_sphericity'] = None

    return pd.DataFrame([features])


# ---------------------------------------------------------------------------
# Batch feature extraction (multiple image stacks)
# ---------------------------------------------------------------------------

_FEATURE_NAMES = [
    'main_tumor_volume', 'main_tumor_volume_mc',
    'main_tumor_surface_area', 'main_tumor_sphericity',
    'second_tumor_volume', 'second_tumor_volume_mc',
    'second_tumor_surface_area', 'second_tumor_sphericity',
    'avg_rest_volume', 'avg_rest_volume_mc',
    'avg_rest_surface_area', 'avg_rest_sphericity',
    'num_of_tumors',
    'total_volume', 'total_volume_mc', 'total_surface_area',
    'median_volume', 'median_volume_mc', 'median_surface_area',
]


def extract_tumor_shape_features(
    dataframes: List[pd.DataFrame],
) -> Tuple:
    """
    Extract shape features from pre-computed measurement DataFrames.

    Parameters
    ----------
    dataframes : list[pd.DataFrame]
        One DataFrame per image stack with columns:
        ``['Volume', 'Volume_MC', 'Surface_Area_MC', 'Sphericity_MC']``.

    Returns
    -------
    tuple
        One list per feature in ``_FEATURE_NAMES`` (same order), each list
        has one entry per input DataFrame.
    """
    feature_lists: Dict[str, list] = {name: [] for name in _FEATURE_NAMES}

    def _metrics(data):
        """Return (vol, vol_mc, sa, sph) from a row or sub-DataFrame."""
        if isinstance(data, pd.Series):
            return (
                data['Volume'],
                data['Volume_MC'],
                data['Surface_Area_MC'],
                data['Sphericity_MC'],
            )
        return (
            data['Volume'].mean(),
            data['Volume_MC'].mean(),
            data['Surface_Area_MC'].mean(),
            data['Sphericity_MC'].mean(),
        )

    for dataframe in dataframes:
        if dataframe.empty:
            for key in feature_lists:
                feature_lists[key].append(None)
            continue

        sorted_df = dataframe.sort_values('Volume', ascending=False).reset_index(drop=True)

        # General statistics
        feature_lists['num_of_tumors'].append(len(sorted_df))
        feature_lists['total_volume'].append(sorted_df['Volume'].sum())
        feature_lists['total_volume_mc'].append(sorted_df['Volume_MC'].sum())
        feature_lists['total_surface_area'].append(sorted_df['Surface_Area_MC'].sum())
        feature_lists['median_volume'].append(sorted_df['Volume'].median())
        feature_lists['median_volume_mc'].append(sorted_df['Volume_MC'].median())
        feature_lists['median_surface_area'].append(sorted_df['Surface_Area_MC'].median())

        # Main tumour
        vol, vol_mc, sa, sph = _metrics(sorted_df.iloc[0])
        feature_lists['main_tumor_volume'].append(vol)
        feature_lists['main_tumor_volume_mc'].append(vol_mc)
        feature_lists['main_tumor_surface_area'].append(sa)
        feature_lists['main_tumor_sphericity'].append(sph)

        # Second largest
        if len(sorted_df) > 1:
            vol, vol_mc, sa, sph = _metrics(sorted_df.iloc[1])
        else:
            vol = vol_mc = sa = sph = None
        feature_lists['second_tumor_volume'].append(vol)
        feature_lists['second_tumor_volume_mc'].append(vol_mc)
        feature_lists['second_tumor_surface_area'].append(sa)
        feature_lists['second_tumor_sphericity'].append(sph)

        # Remaining average
        if len(sorted_df) > 2:
            vol, vol_mc, sa, sph = _metrics(sorted_df.iloc[2:])
        else:
            vol = vol_mc = sa = sph = None
        feature_lists['avg_rest_volume'].append(vol)
        feature_lists['avg_rest_volume_mc'].append(vol_mc)
        feature_lists['avg_rest_surface_area'].append(sa)
        feature_lists['avg_rest_sphericity'].append(sph)

    return tuple(feature_lists.values())


def create_features_dataframe(
    feature_results: Tuple,
    dataset_labels: List[str],
) -> pd.DataFrame:
    """
    Convert the output of ``extract_tumor_shape_features`` into a DataFrame.

    Parameters
    ----------
    feature_results : tuple
        Output of ``extract_tumor_shape_features``.
    dataset_labels : list[str]
        One label per image stack (used as row identifiers).

    Returns
    -------
    pd.DataFrame
        Rows = datasets, columns = shape features.
    """
    features_dict: Dict[str, list] = {'dataset': dataset_labels}
    for i, name in enumerate(_FEATURE_NAMES):
        features_dict[name] = feature_results[i]

    return pd.DataFrame(features_dict)