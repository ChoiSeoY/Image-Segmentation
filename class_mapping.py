import pandas as pd

def load_class_mapping(csv_path):
    """
    Load class mapping from a CSV file.

    Args:
        csv_path (str): Path to the CSV file containing class mappings.

    Returns:
        dict: Mapping of RGB tuples to class IDs.
    """
    df = pd.read_csv(csv_path)
    mapping = {}
    for _, row in df.iterrows():
        rgb = (row['r'], row['g'], row['b'])
        mapping[rgb] = len(mapping)  # Automatically assign class IDs
    return mapping
