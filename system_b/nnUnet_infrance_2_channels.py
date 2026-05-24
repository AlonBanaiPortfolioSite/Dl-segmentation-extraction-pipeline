import subprocess
import nibabel as nib
import numpy as np
from pathlib import Path
import argparse
import shutil

def split_image(image_path_ch0, image_path_ch1, num_splits, tmp_dir):
    """Split a 3D image into k parts along z-axis"""
    print(f"Loading {image_path_ch0.name} and {image_path_ch1.name}...")
    img_ch0 = nib.load(image_path_ch0)
    img_ch1 = nib.load(image_path_ch1)
    
    data_ch0 = img_ch0.get_fdata()
    data_ch1 = img_ch1.get_fdata()
    affine = img_ch0.affine
    
    # Get case name (remove _0000.nii.gz)
    case_name = image_path_ch0.name.replace("_0000.nii.gz", "")
    
    # Calculate split points along z-axis
    z_size = data_ch0.shape[2]
    split_size = z_size // num_splits
    
    chunk_paths = []
    for i in range(num_splits):
        start_z = i * split_size
        end_z = (i + 1) * split_size if i < num_splits - 1 else z_size
        
        chunk_data_ch0 = data_ch0[:, :, start_z:end_z]
        chunk_data_ch1 = data_ch1[:, :, start_z:end_z]
        
        chunk_name_ch0 = f"{case_name}_chunk{i}_0000.nii.gz"
        chunk_name_ch1 = f"{case_name}_chunk{i}_0001.nii.gz"
        
        chunk_path_ch0 = tmp_dir / "chunks" / chunk_name_ch0
        chunk_path_ch1 = tmp_dir / "chunks" / chunk_name_ch1
        
        print(f"  Saving chunk {i}: slices {start_z}:{end_z}")
        
        nib.save(nib.Nifti1Image(chunk_data_ch0, affine), chunk_path_ch0)
        nib.save(nib.Nifti1Image(chunk_data_ch1, affine), chunk_path_ch1)
        
        chunk_paths.append((chunk_path_ch0, chunk_path_ch1, case_name, i))
    
    return chunk_paths, affine

def run_inference(chunks_dir, output_dir, dataset_id, config, fold, trainer):
    """Run nnUNet inference"""
    print("Running nnUNet inference...")
    cmd = [
        "nnUNetv2_predict",
        "-i", str(chunks_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", config,
        "-f", str(fold),
        "-tr", trainer,
        "-npp", "0",
        "-nps", "0"
    ]
    
    subprocess.run(cmd, check=True)
    print("Inference complete!")

def merge_predictions(case_name, num_splits, predictions_dir, output_path, affine):
    """Merge split predictions back together"""
    print(f"Merging predictions for {case_name}...")
    
    chunks = []
    for i in range(num_splits):
        chunk_seg_path = predictions_dir / f"{case_name}_chunk{i}.nii.gz"
        if not chunk_seg_path.exists():
            raise FileNotFoundError(f"Missing prediction: {chunk_seg_path}")
        
        chunk_seg = nib.load(chunk_seg_path).get_fdata()
        chunks.append(chunk_seg)
    
    # Concatenate along z-axis
    merged = np.concatenate(chunks, axis=2)
    
    # Save merged result
    print(f"Saving merged result to {output_path}")
    nib.save(nib.Nifti1Image(merged, affine), output_path)

def main():
    parser = argparse.ArgumentParser(description="Split, predict, and merge nnUNet inference")
    parser.add_argument("--input_folder", type=str, required=True, help="Folder with input images")
    parser.add_argument("--output_folder", type=str, required=True, help="Folder to save final predictions")
    parser.add_argument("--num_splits", type=int, default=2, help="Number of splits per image")
    parser.add_argument("--dataset_id", type=int, required=True, help="nnUNet dataset ID")
    parser.add_argument("--config", type=str, default="2d", help="nnUNet configuration (e.g., 2d, 3d_fullres)")
    parser.add_argument("--fold", type=int, default=0, help="nnUNet fold")
    parser.add_argument("--trainer", type=str, default="nnUNetTrainer", help="Trainer name")
    
    args = parser.parse_args()
    
    # Setup paths
    input_folder = Path(args.input_folder)
    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Create tmp directory
    tmp_dir = Path("tmp")
    chunks_dir = tmp_dir / "chunks"
    predictions_dir = tmp_dir / "predictions"
    
    # Clean and create tmp directories
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    chunks_dir.mkdir(parents=True)
    predictions_dir.mkdir(parents=True)
    
    # Get all input images (channel 0)
    image_files_ch0 = sorted(input_folder.glob("*_0000.nii.gz"))
    print(f"Found {len(image_files_ch0)} images to process")
    
    # Process each image
    for image_path_ch0 in image_files_ch0:
        # Get corresponding channel 1 file
        case_name = image_path_ch0.name.replace("_0000.nii.gz", "")
        image_path_ch1 = input_folder / f"{case_name}_0001.nii.gz"
        
        # Check if channel 1 exists
        if not image_path_ch1.exists():
            print(f"⚠️  Warning: Missing channel 1 for {case_name}, skipping...")
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {case_name}")
        print(f"{'='*60}")
        
        # Split image (both channels)
        chunk_info, affine = split_image(image_path_ch0, image_path_ch1, args.num_splits, tmp_dir)
        case_name = chunk_info[0][2]
        
        # Run inference
        run_inference(chunks_dir, predictions_dir, args.dataset_id, args.config, args.fold, args.trainer)
        
        # Merge predictions
        output_path = output_folder / f"{case_name}.nii.gz"
        merge_predictions(case_name, args.num_splits, predictions_dir, output_path, affine)
        
        # Clean tmp for next image
        for f in chunks_dir.glob("*"):
            f.unlink()
        for f in predictions_dir.glob("*"):
            f.unlink()
        
        print(f"✓ Completed: {case_name}")
    
    print(f"\n{'='*60}")
    print(f"All images processed! Results saved to: {output_folder}")
    print(f"{'='*60}")
    
    # Cleanup tmp directory
    shutil.rmtree(tmp_dir)

print("Script started!")
if __name__ == "__main__":
    print("Main function called!")
    main()