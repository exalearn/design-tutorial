"""Features that describe an entire graph"""

from networkx.linalg import spectrum
import numpy as np


def compute_features(graph, spectra_fun=spectrum.laplacian_spectrum, target_length=32):
    """Compute the features for a graph
    
    Args:
        graph (networkx): Graph to featurize
        spectra_fun (Callable): Method to use to compute spectra
        target_length (int): Number of features to include
    """
    
    # Compute the spectrum
    spectrum = spectra_fun(graph)
    
    # Force it to be an array of doubles, sorted in decreasing order
    spectrum = np.absolute(spectrum, dtype=np.float).flatten()
    spectrum = np.sort(spectrum)[::-1]
    
    # Return the desired number of elements
    if spectrum.size > target_length:
        return spectrum[:target_length]
    else:
        output = np.zeros((target_length,))
        output[:spectrum.size] = spectrum    
        return output
