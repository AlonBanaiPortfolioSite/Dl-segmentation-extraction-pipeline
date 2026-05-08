"""
GMM Segmentation (Experimental)
================================
Gaussian Mixture Model based foreground/background segmentation.

.. note::
    This module was explored during pipeline development but is **not used
    in the final segmentation pipeline**.  It is retained for
    reproducibility and potential future use.
"""

import numpy as np
from sklearn.mixture import GaussianMixture
from typing import Tuple


def gmm_segment_slice(
    image: np.ndarray,
    means_init: np.ndarray,
    weights_init: np.ndarray,
    n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment a 2D image into *n_components* classes using a Gaussian Mixture
    Model with user-supplied initialisation.

    Parameters
    ----------
    image : np.ndarray
        2D array (Y, X) of intensity values.
    means_init : array-like, shape (n_components,)
        Initial mean for each component.
    weights_init : array-like, shape (n_components,)
        Initial mixing weights (must sum to 1).
    n_components : int
        Number of Gaussian components.

    Returns
    -------
    means : np.ndarray, shape (n_components, 1)
        Fitted component means.
    weights : np.ndarray, shape (n_components,)
        Fitted mixing weights.
    labels : np.ndarray, shape (Y, X)
        Per-pixel component assignment, same spatial shape as *image*.
    """
    pixel_vector = image.reshape(-1, 1)

    gmm = GaussianMixture(
        n_components=n_components,
        means_init=np.asarray(means_init).reshape(n_components, 1),
        weights_init=np.asarray(weights_init),
    )
    gmm.fit(pixel_vector)

    labels = gmm.predict(pixel_vector).reshape(image.shape)
    return gmm.means_, gmm.weights_, labels


def gmm_segment_stack_sequential(
    stack: np.ndarray,
    means_init: np.ndarray,
    weights_init: np.ndarray,
    n_components: int = 2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Segment a 3D stack slice-by-slice, propagating the fitted parameters
    of slice *i* as the initialisation for slice *i + 1*.

    This encourages temporal/spatial consistency across slices by warm-
    starting each fit from a nearby solution.

    Parameters
    ----------
    stack : np.ndarray
        3D array of shape ``(Z, Y, X)``.
    means_init : array-like, shape (n_components,)
        Initial means for the first slice.
    weights_init : array-like, shape (n_components,)
        Initial weights for the first slice (must sum to 1).
    n_components : int
        Number of Gaussian components.

    Returns
    -------
    all_means : np.ndarray, shape (Z, n_components, 1)
        Fitted means per slice.
    all_weights : np.ndarray, shape (Z, n_components)
        Fitted weights per slice.
    label_stack : np.ndarray, shape (Z, Y, X)
        Per-voxel component assignment.
    """
    depth = stack.shape[0]
    label_stack = np.zeros(stack.shape, dtype=int)
    all_means = np.zeros((depth, n_components, 1))
    all_weights = np.zeros((depth, n_components))

    current_means = np.asarray(means_init)
    current_weights = np.asarray(weights_init)

    for i in range(depth):
        means, weights, labels = gmm_segment_slice(
            stack[i],
            means_init=current_means,
            weights_init=current_weights,
            n_components=n_components,
        )
        label_stack[i] = labels
        all_means[i] = means
        all_weights[i] = weights

        # Propagate fitted parameters to next slice
        current_means = means.ravel()
        current_weights = weights

    return all_means, all_weights, label_stack