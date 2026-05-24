

"""
Registration module for ANTsPy-based image registration.
Rigid registration with cross-correlation metric.
"""

import ants
import numpy as np


def register(img_fixed, img_moving, 
             mask=None,
             aff_sampling=12,
             aff_iterations=(500, 250, 100, 20),
             verbose=True):
    """
    Register moving image to fixed image using rigid transformation and CC metric.
    
    Parameters:
    -----------
    img_fixed : numpy.ndarray or ants.ANTsImage
        Fixed/reference image (3D array)
    img_moving : numpy.ndarray or ants.ANTsImage
        Moving image to be registered (3D array)
    mask : numpy.ndarray or ants.ANTsImage, optional
        Binary mask for registration (applied to fixed image)
    aff_sampling : int
        Sampling rate for registration (higher = faster but less accurate)
        Default: 12
    aff_iterations : tuple
        Number of iterations at each resolution level
        Default: (500, 250, 100, 20) - from coarse to fine
    verbose : bool
        Print detailed registration progress
    
    Returns:
    --------
    dict with keys:
        - 'registered_image': ants.ANTsImage - The registered moving image
        - 'registered_array': numpy.ndarray - The registered image as numpy array
        - 'forward_transforms': list - Forward transform files
        - 'inverse_transforms': list - Inverse transform files
    """
    
    # Convert to ANTs format if needed
    if isinstance(img_fixed, np.ndarray):
        if verbose:
            print(f"Converting fixed image to ANTs format: shape {img_fixed.shape}")
        fixed_ants = ants.from_numpy(img_fixed.astype(np.float32))
    else:
        fixed_ants = img_fixed
    
    if isinstance(img_moving, np.ndarray):
        if verbose:
            print(f"Converting moving image to ANTs format: shape {img_moving.shape}")
        moving_ants = ants.from_numpy(img_moving.astype(np.float32))
    else:
        moving_ants = img_moving
    
    # Convert mask to ANTs format if provided
    if mask is not None:
        if isinstance(mask, np.ndarray):
            if verbose:
                print(f"Converting mask to ANTs format: shape {mask.shape}")
            mask_ants = ants.from_numpy(mask.astype(np.float32))
        else:
            mask_ants = mask
    else:
        mask_ants = None
    
    # Print registration parameters
    if verbose:
        print(f"\nRegistration parameters:")
        print(f"  Transform type: Rigid")
        print(f"  Metric: CC (Cross-Correlation)")
        print(f"  Sampling: {aff_sampling}")
        print(f"  Iterations: {aff_iterations}")
        print(f"  Using mask: {mask_ants is not None}")
    
    # Run registration
    if verbose:
        print(f"\nStarting registration...")
    
    registration_result = ants.registration(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform='Rigid',
        mask=mask_ants,
        aff_metric='CC',
        aff_sampling=aff_sampling,
        aff_iterations=aff_iterations,
        verbose=verbose
    )
    
    if verbose:
        print(f"Registration completed!")
    
    # Extract registered image
    registered_image = registration_result['warpedmovout']
    registered_array = registered_image.numpy()
    
    # Prepare return dictionary
    result = {
        'registered_image': registered_image,
        'registered_array': registered_array,
        'forward_transforms': registration_result.get('fwdtransforms', []),
        'inverse_transforms': registration_result.get('invtransforms', []),
        'full_result': registration_result
    }
    
    if verbose:
        print(f"Registered image shape: {registered_array.shape}")
        print(f"Number of transforms: {len(result['forward_transforms'])}")
    
    return result


def apply_transforms(img_to_transform, fixed_reference, transform_list, 
                     interpolation='linear', verbose=False):
    """
    Apply existing transforms to a new image.
    
    Useful for applying registration transforms to additional channels
    or segmentation masks.
    
    Parameters:
    -----------
    img_to_transform : numpy.ndarray or ants.ANTsImage
        Image to transform
    fixed_reference : numpy.ndarray or ants.ANTsImage
        Reference image (defines output space)
    transform_list : list
        List of transform files (from registration result)
    interpolation : str
        Interpolation method: 'linear', 'nearestNeighbor', 'bSpline'
    verbose : bool
        Print progress information
    
    Returns:
    --------
    dict with keys:
        - 'transformed_image': ants.ANTsImage
        - 'transformed_array': numpy.ndarray
    """
    
    # Convert to ANTs format if needed
    if isinstance(img_to_transform, np.ndarray):
        if verbose:
            print(f"Converting image to ANTs format: shape {img_to_transform.shape}")
        img_ants = ants.from_numpy(img_to_transform.astype(np.float32))
    else:
        img_ants = img_to_transform
    
    if isinstance(fixed_reference, np.ndarray):
        ref_ants = ants.from_numpy(fixed_reference.astype(np.float32))
    else:
        ref_ants = fixed_reference
    
    if verbose:
        print(f"Applying {len(transform_list)} transforms with {interpolation} interpolation...")
    
    # Apply transforms
    transformed = ants.apply_transforms(
        fixed=ref_ants,
        moving=img_ants,
        transformlist=transform_list,
        interpolator=interpolation
    )
    
    if verbose:
        print(f"Transform application completed!")
    
    return {
        'transformed_image': transformed,
        'transformed_array': transformed.numpy()
    }