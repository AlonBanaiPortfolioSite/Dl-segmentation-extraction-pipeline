"""
nnUNet format conversion utilities.
Converts multi-channel images to nnUNet's expected format.
"""

import numpy as np
import nibabel as nib
from pathlib import Path
import json


def convert_to_nnunet_format(image, 
                             spacing, 
                             output_path,
                             case_id,
                             dataset_name="Dataset001_OrganoidTumor"):
    """
    Convert multi-channel image to nnUNet format.
    
    nnUNet expects:
    - Images as NIfTI files (.nii.gz)
    - Multi-channel images with suffix: _0000.nii.gz, _0001.nii.gz, etc.
    - Naming: {case_id}_{channel_idx:04d}.nii.gz
    - Example: organoid_001_0000.nii.gz, organoid_001_0001.nii.gz
    
    Parameters:
    -----------
    image : numpy.ndarray
        Image array with shape (n_channels, Z, Y, X)
    spacing : tuple
        Voxel spacing in (z, y, x) order in microns
    output_path : str or Path
        Base output directory (will create nnUNet structure inside)
    case_id : str
        Case identifier (e.g., "organoid_001")
    dataset_name : str
        Dataset name (default: "Dataset001_OrganoidTumor")
    
    Returns:
    --------
    dict with paths to saved files
    """
    
    output_path = Path(output_path)
    
    # Create nnUNet directory structure
    # nnUNet expects: nnUNet_raw/DatasetXXX_Name/imagesTr/
    dataset_path = output_path / "nnUNet_raw" / dataset_name / "imagesTr"
    dataset_path.mkdir(parents=True, exist_ok=True)
    
    # Convert spacing from microns to mm (nnUNet uses mm)
    spacing_mm =spacing
    
    # Ensure image has correct shape
    if image.ndim == 3:
        # Single channel, add channel dimension
        image = image[np.newaxis, ...]
    
    n_channels = image.shape[0]
    
    print(f"Converting to nnUNet format:")
    print(f"  Case ID: {case_id}")
    print(f"  Number of channels: {n_channels}")
    print(f"  Image shape per channel: {image.shape[1:]}")
    print(f"  Spacing (mm): {spacing_mm}")
    
    saved_files = []
    
    # Save each channel separately
    for ch_idx in range(n_channels):
        # nnUNet naming: {case_id}_{channel_idx:04d}.nii.gz
        filename = f"{case_id}_{ch_idx:04d}.nii.gz"
        filepath = dataset_path / filename
        
        # Get channel data
        channel_data = image[ch_idx, ...]
        # Transpose from (Z, Y, X) to (X, Y, Z) for nnUNet
        channel_data = np.transpose(channel_data, (2, 1, 0))  # ← ADD THIS LINE at 13\1\26
        # Create affine matrix with spacing
        # nnUNet uses RAS+ orientation
        affine = np.eye(4)
        affine[0, 0] = spacing_mm[2]  # X spacing
        affine[1, 1] = spacing_mm[1]  # Y spacing
        affine[2, 2] = spacing_mm[0]  # Z spacing
        
        # Create NIfTI image
        nifti_img = nib.Nifti1Image(channel_data, affine)
        
        # Save
        nib.save(nifti_img, str(filepath))
        saved_files.append(str(filepath))
        
        print(f"  Saved channel {ch_idx}: {filename}")
    
    # Create dataset.json if it doesn't exist
    dataset_json_path = output_path / "nnUNet_raw" / dataset_name / "dataset.json"
    if not dataset_json_path.exists():
        create_nnunet_dataset_json(
            output_path=dataset_json_path,
            dataset_name=dataset_name,
            n_channels=n_channels
        )
    
    return {
        'saved_files': saved_files,
        'dataset_path': str(dataset_path),
        'case_id': case_id,
        'n_channels': n_channels
    }


def create_nnunet_dataset_json(output_path, dataset_name, n_channels=2):
    """
    Create dataset.json file required by nnUNet.
    
    Parameters:
    -----------
    output_path : str or Path
        Path to save dataset.json
    dataset_name : str
        Name of the dataset
    n_channels : int
        Number of input channels
    """
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Channel names (you can customize these)
    if n_channels == 2:
        channel_names = {
            "0": "tumor",
            "1": "organoid"
        }
    else:
        channel_names = {str(i): f"channel_{i}" for i in range(n_channels)}
    
    dataset_json = {
        "name": dataset_name,
        "description": "Organoid and tumor registration dataset",
        "reference": "Your institution/paper",
        "licence": "Your license",
        "release": "1.0",
        "tensorImageSize": "4D",
        "modality": channel_names,
        "labels": {
            "background": 0,
            "tumor": 1,
            "organoid": 2
        },
        "numTraining": 0,  # Will be updated as you add cases
        "file_ending": ".nii.gz"
    }
    
    with open(output_path, 'w') as f:
        json.dump(dataset_json, f, indent=4)
    
    print(f"Created dataset.json: {output_path}")
    
    return dataset_json


def convert_tiff_to_nnunet(tiff_path,
                           output_path,
                           case_id,
                           tumor_channel=1,
                           organoid_channel=0,
                           register_first=True,
                           percentile_th=80):
    """
    Complete workflow: Load TIFF, optionally register, convert to nnUNet.
    
    Parameters:
    -----------
    tiff_path : str or Path
        Path to input TIFF file
    output_path : str or Path
        Output directory for nnUNet format
    case_id : str
        Case identifier
    tumor_channel : int
        Tumor channel index
    organoid_channel : int
        Organoid channel index
    register_first : bool
        Whether to register organoid to tumor before conversion
    percentile_th : int
        Percentile threshold for tumor mask (if register_first=True)
    
    Returns:
    --------
    dict with conversion results
    """
    
    from loading_and_saving_images import load_image_stack, get_resolution_from_tiff
    
    print(f"Loading TIFF: {tiff_path}")
    
    # Load images
    tumor = load_image_stack(tiff_path, channel_num=tumor_channel)
    organoid = load_image_stack(tiff_path, channel_num=organoid_channel)
    
    # Get spacing
    x_microns, y_microns, z_microns = get_resolution_from_tiff(tiff_path)
    spacing = (z_microns, y_microns, x_microns)
    
    # Register if requested
    if register_first:
        print("Registering organoid to tumor...")
        from registration import register
        
        th = np.percentile(tumor, percentile_th)
        tumor_mask = tumor > th
        
        result = register(
            img_fixed=tumor,
            img_moving=organoid,
            mask=tumor_mask,
            verbose=True
        )
        
        organoid_registered = result['registered_array']
    else:
        organoid_registered = organoid
    
    # Stack channels: tumor=0, organoid=1
    combined_image = np.stack([tumor, organoid_registered], axis=0)
    
    print(f"Combined image shape: {combined_image.shape}")
    print(f"Spacing: {spacing} microns")
    
    # Convert to nnUNet format
    result = convert_to_nnunet_format(
        image=combined_image,
        spacing=spacing,
        output_path=output_path,
        case_id=case_id
    )
    
    return result
