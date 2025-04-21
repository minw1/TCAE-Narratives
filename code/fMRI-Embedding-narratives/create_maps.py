import nibabel.freesurfer.io as fsio
import numpy as np

def load_annot_to_binary_dict(lh_annot_file, rh_annot_file):
    """
    Load left and right hemisphere .annot files and create a dictionary of binary maps for regions.

    Parameters:
    - lh_annot_file: Path to the left hemisphere .annot file.
    - rh_annot_file: Path to the right hemisphere .annot file.

    Returns:
    - region_dict: Dictionary where keys are region names (prefixed with L_ or R_) and values are binary arrays.
    """
    region_dict = {}

    # Process left hemisphere
    labels, _, names = fsio.read_annot(lh_annot_file)
    for i, name in enumerate(names):
        region_name = f"L_{name.decode('utf-8')}"
        binary_map = (labels == i).astype(int)
        dummy_map = np.zeros_like(binary_map)
        region_dict[region_name] = np.hstack((binary_map, dummy_map))

    # Process right hemisphere
    labels, _, names = fsio.read_annot(rh_annot_file)
    for i, name in enumerate(names):
        region_name = f"R_{name.decode('utf-8')}"
        binary_map = (labels == i).astype(int)
        dummy_map = np.zeros_like(binary_map)
        region_dict[region_name] = np.hstack((dummy_map, binary_map))

    return region_dict
