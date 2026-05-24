"""
Two-Model Segmentation Pipeline
================================
Runs two single-class nnU-Net models (tumor=302, organoid=304) independently
and merges their outputs into a unified two-class segmentation mask.

Pipeline per case:
1. Run tumor model (Dataset302) on the tumor channel (_0001) → tumor mask
2. Zero out tumor-positive pixels on the organoid channel (_0000)
3. Run organoid model (Dataset304) on the cleaned organoid channel → organoid mask
4. Merge masks: background=0, organoid=1, tumor=2
   - Pixels positive in BOTH masks → tumor (class 2)

Usage:
    python two_model_inference.py \
        --input_folder /path/to/input \
        --base_output /path/to/2_models_segmentation \
        --num_splits 5 \
        --config 2d \
        --tumor_fold 0 \
        --organoid_fold 0
"""

import subprocess
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import shutil


def split_image(image_path, num_splits, chunks_dir, case_name):
    """Split a 3D NIfTI image into chunks along z-axis for memory-safe inference.

    Parameters
    ----------
    image_path : Path
        Path to the input .nii.gz file.
    num_splits : int
        Number of chunks to split the volume into.
    chunks_dir : Path
        Directory to save chunk files.
    case_name : str
        Base name for chunk files (without channel suffix).

    Returns
    -------
    affine : np.ndarray
        The affine matrix from the original image.
    z_size : int
        Total number of slices along z-axis.
    """
    print(f"  Loading {image_path.name}...")
    img = nib.load(image_path)
    data = img.get_fdata()
    affine = img.affine
    z_size = data.shape[2]
    split_size = z_size // num_splits

    for i in range(num_splits):
        start_z = i * split_size
        end_z = (i + 1) * split_size if i < num_splits - 1 else z_size
        chunk_data = data[:, :, start_z:end_z]
        # Single-channel models expect _0000 suffix
        chunk_path = chunks_dir / f"{case_name}_chunk{i}_0000.nii.gz"
        print(f"    Chunk {i}: slices {start_z}:{end_z}")
        nib.save(nib.Nifti1Image(chunk_data, affine), chunk_path)

    return affine, z_size


def run_inference(chunks_dir, predictions_dir, dataset_id, config, fold,
                  trainer="nnUNetTrainer"):
    """Run nnU-Net inference on a folder of chunks.

    Parameters
    ----------
    chunks_dir : Path
        Directory containing input chunk files.
    predictions_dir : Path
        Directory where nnU-Net writes prediction files.
    dataset_id : int
        nnU-Net dataset ID (302 for tumor, 304 for organoid).
    config : str
        nnU-Net configuration (e.g., '2d').
    fold : int
        Model fold to use.
    trainer : str
        nnU-Net trainer name.
    """
    print(f"  Running nnU-Net inference (Dataset {dataset_id}, fold {fold}, "
          f"trainer {trainer})...")
    cmd = [
        "nnUNetv2_predict",
        "-i", str(chunks_dir),
        "-o", str(predictions_dir),
        "-d", str(dataset_id),
        "-c", config,
        "-f", str(fold),
        "-tr", trainer,
        "-npp", "0",
        "-nps", "0",
    ]
    subprocess.run(cmd, check=True)
    print("  Inference complete.")


def merge_chunks(case_name, num_splits, predictions_dir, affine, z_size):
    """Merge chunked predictions back into a single 3D volume.

    Parameters
    ----------
    case_name : str
        Base case name used when splitting.
    num_splits : int
        Number of chunks that were created.
    predictions_dir : Path
        Directory containing prediction chunk files.
    affine : np.ndarray
        Original affine matrix for the output NIfTI.
    z_size : int
        Expected total z-dimension (for validation).

    Returns
    -------
    merged : np.ndarray
        Merged 3D segmentation mask.
    """
    chunks = []
    for i in range(num_splits):
        chunk_path = predictions_dir / f"{case_name}_chunk{i}.nii.gz"
        if not chunk_path.exists():
            raise FileNotFoundError(f"Missing prediction chunk: {chunk_path}")
        chunk_data = nib.load(chunk_path).get_fdata()
        chunks.append(chunk_data)

    merged = np.concatenate(chunks, axis=2)
    assert merged.shape[2] == z_size, (
        f"Merged z-size {merged.shape[2]} != expected {z_size}"
    )
    return merged


def clean_tmp(directory):
    """Remove all files inside a directory without deleting the directory."""
    for f in Path(directory).glob("*"):
        if f.is_file():
            f.unlink()


def process_case(subfolder, args, tmp_dir):
    """Run the full two-model pipeline on a single case subfolder.

    Parameters
    ----------
    subfolder : Path
        Directory containing {case}_0000.nii.gz and {case}_0001.nii.gz.
    args : argparse.Namespace
        Command-line arguments.
    tmp_dir : Path
        Temporary working directory.

    Returns
    -------
    case_name : str
        The case identifier extracted from filenames.
    """
    chunks_dir = tmp_dir / "chunks"
    predictions_dir = tmp_dir / "predictions"

    # Find input files
    organoid_files = sorted(subfolder.glob("*_0000.nii.gz"))
    tumor_files = sorted(subfolder.glob("*_0001.nii.gz"))

    if len(organoid_files) != 1 or len(tumor_files) != 1:
        raise ValueError(
            f"Expected exactly 1 _0000 and 1 _0001 file in {subfolder}, "
            f"found {len(organoid_files)} and {len(tumor_files)}"
        )

    organoid_path = organoid_files[0]
    tumor_path = tumor_files[0]
    # Extract case name: e.g., "p74" from "p74_0001.nii.gz"
    case_name = tumor_path.name.replace("_0001.nii.gz", "")

    print(f"\n{'=' * 60}")
    print(f"Processing: {case_name}")
    print(f"  Organoid channel: {organoid_path.name}")
    print(f"  Tumor channel:    {tumor_path.name}")
    print(f"{'=' * 60}")

    # ------------------------------------------------------------------
    # STEP 1: Run tumor model (302) on tumor channel
    # ------------------------------------------------------------------
    print("\n[STEP 1] Tumor segmentation (Dataset 302)")
    clean_tmp(chunks_dir)
    clean_tmp(predictions_dir)

    tumor_affine, tumor_z = split_image(
        tumor_path, args.num_splits, chunks_dir, case_name
    )
    run_inference(
        chunks_dir, predictions_dir,
        dataset_id=302, config=args.config, fold=args.tumor_fold,
        trainer=args.tumor_trainer,
    )
    tumor_mask = merge_chunks(
        case_name, args.num_splits, predictions_dir, tumor_affine, tumor_z
    )
    # Binarize (in case of soft values)
    tumor_mask_binary = (tumor_mask > 0).astype(np.uint8)

    # Save tumor mask
    tumor_out = args.tumor_output / f"{case_name}.nii.gz"
    nib.save(nib.Nifti1Image(tumor_mask_binary, tumor_affine), tumor_out)
    print(f"  Tumor mask saved: {tumor_out}")
    print(f"  Tumor foreground voxels: {tumor_mask_binary.sum()}")

    # ------------------------------------------------------------------
    # STEP 2: Zero tumor pixels on organoid channel, then run organoid model
    # ------------------------------------------------------------------
    print("\n[STEP 2] Organoid segmentation (Dataset 304)")
    print("  Zeroing tumor pixels on organoid channel...")

    organoid_img = nib.load(organoid_path)
    organoid_data = organoid_img.get_fdata().copy()
    organoid_affine = organoid_img.affine

    # Zero out where tumor was detected
    pixels_zeroed = tumor_mask_binary.sum()
    organoid_data[tumor_mask_binary > 0] = 0
    print(f"  Zeroed {pixels_zeroed} pixels")

    # Save cleaned organoid as chunks for inference
    clean_tmp(chunks_dir)
    clean_tmp(predictions_dir)

    # Save cleaned organoid to a temp nii.gz, then split it
    cleaned_organoid_path = tmp_dir / f"{case_name}_cleaned_organoid.nii.gz"
    nib.save(nib.Nifti1Image(organoid_data, organoid_affine),
             cleaned_organoid_path)

    organoid_affine_out, organoid_z = split_image(
        cleaned_organoid_path, args.num_splits, chunks_dir, case_name
    )
    run_inference(
        chunks_dir, predictions_dir,
        dataset_id=304, config=args.config, fold=args.organoid_fold,
        trainer=args.organoid_trainer,
    )
    organoid_mask = merge_chunks(
        case_name, args.num_splits, predictions_dir,
        organoid_affine_out, organoid_z
    )
    organoid_mask_binary = (organoid_mask > 0).astype(np.uint8)

    # Save organoid mask
    organoid_out = args.organoid_output / f"{case_name}.nii.gz"
    nib.save(nib.Nifti1Image(organoid_mask_binary, organoid_affine),
             organoid_out)
    print(f"  Organoid mask saved: {organoid_out}")
    print(f"  Organoid foreground voxels: {organoid_mask_binary.sum()}")

    # ------------------------------------------------------------------
    # STEP 3: Merge into two-class mask
    # ------------------------------------------------------------------
    print("\n[STEP 3] Merging into two-class mask")
    # Start with background (0)
    merged = np.zeros_like(tumor_mask_binary, dtype=np.uint8)

    # Organoid = class 1 (invading cells / microglia)
    merged[organoid_mask_binary > 0] = 1

    # Tumor = class 2 — overwrites organoid where both are positive
    merged[tumor_mask_binary > 0] = 2

    # Summary
    n_bg = (merged == 0).sum()
    n_organoid = (merged == 1).sum()
    n_tumor = (merged == 2).sum()
    n_overlap = ((tumor_mask_binary > 0) & (organoid_mask_binary > 0)).sum()
    print(f"  Background (0): {n_bg}")
    print(f"  Organoid   (1): {n_organoid}")
    print(f"  Tumor      (2): {n_tumor}")
    print(f"  Overlap (both FG → marked tumor): {n_overlap}")

    merged_out = args.both_output / f"{case_name}.nii.gz"
    nib.save(nib.Nifti1Image(merged, tumor_affine), merged_out)
    print(f"  Merged mask saved: {merged_out}")

    # Clean temp files (keep folder)
    clean_tmp(chunks_dir)
    clean_tmp(predictions_dir)
    # Also remove the temp cleaned organoid file
    if cleaned_organoid_path.exists():
        cleaned_organoid_path.unlink()

    return case_name


def main():
    parser = argparse.ArgumentParser(
        description="Two-model segmentation: tumor (302) + organoid (304) → merged mask"
    )
    parser.add_argument(
        "--input_folder", type=str, required=True,
        help="Parent folder containing per-case subfolders, each with "
             "_0000.nii.gz and _0001.nii.gz"
    )
    parser.add_argument(
        "--base_output", type=str,
        default="/home/alon.banai@bm.technion.ac.il/Alon/test_registration/"
                "2_models_segmentation",
        help="Base output directory (will contain tumor/, organoid/, both/, tmp/)"
    )
    parser.add_argument("--num_splits", type=int, default=5,
                        help="Number of z-axis chunks per volume")
    parser.add_argument("--config", type=str, default="2d",
                        help="nnU-Net configuration")
    parser.add_argument("--tumor_fold", type=int, default=0,
                        help="Fold for tumor model (Dataset 302)")
    parser.add_argument("--organoid_fold", type=int, default=0,
                        help="Fold for organoid model (Dataset 304)")
    parser.add_argument("--tumor_trainer", type=str, default="nnUNetTrainer",
                        help="Trainer for tumor model")
    parser.add_argument("--organoid_trainer", type=str, default="nnUNetTrainer",
                        help="Trainer for organoid model")

    args = parser.parse_args()

    # Setup output directories
    base = Path(args.base_output)
    args.tumor_output = base / "tumor"
    args.organoid_output = base / "organoid"
    args.both_output = base / "both"
    tmp_dir = base / "tmp"

    for d in [args.tumor_output, args.organoid_output, args.both_output, tmp_dir]:
        d.mkdir(parents=True, exist_ok=True)

    # Create chunk subdirs inside tmp
    chunks_dir = tmp_dir / "chunks"
    predictions_dir = tmp_dir / "predictions"
    chunks_dir.mkdir(exist_ok=True)
    predictions_dir.mkdir(exist_ok=True)

    # Find all case subfolders
    input_folder = Path(args.input_folder)
    subfolders = sorted([
        d for d in input_folder.iterdir()
        if d.is_dir() and list(d.glob("*_0000.nii.gz"))
    ])

    if not subfolders:
        print(f"ERROR: No subfolders with _0000.nii.gz found in {input_folder}")
        print("Expected structure: input_folder/case_name/case_name_0000.nii.gz")
        return

    print(f"Found {len(subfolders)} cases to process:")
    for sf in subfolders:
        print(f"  - {sf.name}")

    # Process each case
    completed = []
    failed = []
    for subfolder in subfolders:
        try:
            case_name = process_case(subfolder, args, tmp_dir)
            completed.append(case_name)
            print(f"\n✓ Completed: {case_name}")
        except Exception as e:
            print(f"\n✗ FAILED: {subfolder.name} — {e}")
            failed.append(subfolder.name)
            # Clean tmp for next case
            clean_tmp(chunks_dir)
            clean_tmp(predictions_dir)

    # Final summary
    print(f"\n{'=' * 60}")
    print(f"DONE — {len(completed)} completed, {len(failed)} failed")
    print(f"  Tumor masks:    {args.tumor_output}")
    print(f"  Organoid masks: {args.organoid_output}")
    print(f"  Merged masks:   {args.both_output}")
    if failed:
        print(f"  Failed cases:   {failed}")
    print(f"{'=' * 60}")

    # Clean tmp contents but keep the folder
    clean_tmp(chunks_dir)
    clean_tmp(predictions_dir)


if __name__ == "__main__":
    main()