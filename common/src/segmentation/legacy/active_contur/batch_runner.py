"""
Batch Experiment Runner
=======================
Run active contour refinement across multiple image/mask pairs with
automatic format handling (TIFF and NIfTI).
"""

import os
from typing import List

import imageio.v3 as iio
import numpy as np

from .active_contour_refinement import run_active_contour_experiment
from .config import (
    ActiveContourConfig,
    ExperimentConfig,
    FilterConfig,
    OutputConfig,
)


def run_batch_experiments(
    experiments_config: List[dict],
    verbose: bool = True,
) -> List[dict]:
    """
    Run active contour experiments on multiple files.

    Parameters
    ----------
    experiments_config : list[dict]
        Each dictionary should contain:

        - ``experiment_name`` : str
        - ``image_path`` : str
        - ``mask_path`` : str
        - ``output_folder`` : str
        - ``file_format`` : ``'tiff'`` or ``'nifti'`` (default ``'tiff'``)
        - ``axis_order`` : ``'zyx'`` or ``'xyz'`` (default ``'zyx'``)
        - ``filter_type`` : ``'gaussian'`` or ``'bilateral'``
        - ``filter_params`` : dict (e.g. ``{'sigma': 3.0}``)
        - ``active_contour_params`` : dict (e.g. ``{'alpha': 0.015}``)

    verbose : bool
        Print detailed progress.

    Returns
    -------
    list[dict]
        Results from each experiment.
    """
    all_results: List[dict] = []

    for i, exp in enumerate(experiments_config):
        print("\n" + "=" * 80)
        print(f"EXPERIMENT {i + 1}/{len(experiments_config)}: {exp['experiment_name']}")
        print("=" * 80)

        file_format = exp.get("file_format", "tiff")
        axis_order = exp.get("axis_order", "zyx")

        # -- Load ----------------------------------------------------------
        print(f"Loading {file_format} files ...")
        print(f"  Image: {exp['image_path']}")
        print(f"  Mask : {exp['mask_path']}")

        if file_format == "tiff":
            image_stack = np.asarray(iio.imread(exp["image_path"]))
            mask_stack = np.asarray(iio.imread(exp["mask_path"]))
        elif file_format == "nifti":
            import nibabel as nib
            image_nifti = nib.load(exp["image_path"])
            mask_nifti = nib.load(exp["mask_path"])
            image_stack = image_nifti.get_fdata()
            mask_stack = mask_nifti.get_fdata()
        else:
            raise ValueError(f"Unknown file format: {file_format}")

        print(f"  Loaded shape: {image_stack.shape}")

        # -- Axis order ----------------------------------------------------
        needs_transpose = False
        if axis_order == "xyz":
            print("  Converting (X, Y, Z) -> (Z, Y, X) ...")
            image_stack = np.transpose(image_stack, (2, 1, 0))
            mask_stack = np.transpose(mask_stack, (2, 1, 0))
            needs_transpose = True
            print(f"  New shape: {image_stack.shape}")

        if mask_stack.max() > 1:
            mask_stack = (mask_stack > 0).astype(np.uint8)

        # -- Config --------------------------------------------------------
        filter_type = exp.get("filter_type", "gaussian")
        fp = exp.get("filter_params", {})

        if filter_type == "gaussian":
            filter_cfg = FilterConfig(filter_type="gaussian", sigma=fp.get("sigma", 3.0))
        else:
            filter_cfg = FilterConfig(
                filter_type="bilateral",
                d=fp.get("d", 7),
                sigma_color=fp.get("sigma_color", 30),
                sigma_space=fp.get("sigma_space", 5),
            )

        acp = exp.get("active_contour_params", {})
        ac_cfg = ActiveContourConfig(
            alpha=acp.get("alpha", 0.015),
            beta=acp.get("beta", 10.0),
            gamma=acp.get("gamma", 0.001),
        )

        config = ExperimentConfig(
            experiment_name=exp["experiment_name"],
            base_folder=exp["output_folder"],
            filter=filter_cfg,
            active_contour=ac_cfg,
            output=OutputConfig(),
        )

        # -- Run -----------------------------------------------------------
        results = run_active_contour_experiment(
            image_stack=image_stack,
            init_contour_stack=mask_stack,
            config=config,
            verbose=verbose,
        )

        # -- Re-transpose & save NIfTI if needed ---------------------------
        if needs_transpose:
            print("Converting results back to (X, Y, Z) ...")
            for key in ("denoised_stack", "init_contour_perimeter", "final_contour_filled"):
                results[key] = np.transpose(results[key], (2, 1, 0))

            if file_format == "nifti":
                import nibabel as nib  # noqa: F811 (re-import for clarity)

                affine = image_nifti.affine  # type: ignore[union-attr]
                out = exp["output_folder"]
                os.makedirs(out, exist_ok=True)

                nib.save(
                    nib.Nifti1Image(results["denoised_stack"].astype(np.float32), affine),
                    os.path.join(out, "denoised_image.nii.gz"),
                )
                nib.save(
                    nib.Nifti1Image(results["final_contour_filled"].astype(np.uint8), affine),
                    os.path.join(out, "final_contour_filled.nii.gz"),
                )
                nib.save(
                    nib.Nifti1Image(results["init_contour_perimeter"].astype(np.uint8), affine),
                    os.path.join(out, "init_contour_perimeter.nii.gz"),
                )
                print(f"  Saved NIfTI files to: {out}")

        all_results.append(results)
        print(f"\nExperiment {i + 1} complete.")

    print("\n" + "=" * 80)
    print(f"ALL {len(experiments_config)} EXPERIMENTS COMPLETE")
    print("=" * 80)

    return all_results