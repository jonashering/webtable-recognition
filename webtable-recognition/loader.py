import os
from sklearn.datasets import load_files
from pandas import DataFrame


def load_from_directory(container_path):
    """
    Load text files with categories as subfolder names to dataframe

    Args:
        container_path: Path to the main folder holding one subfolder per category
    Returns:
        Dataframe containing raw file in raw column and true label in label column
    """
    if not os.path.isdir(container_path):
        raise NotADirectoryError(container_path)

    dataset = load_files(container_path)

    label_names = dataset['target_names']
    raw = dataset['data']
    labels = labels = [label_names[idx] for idx in dataset['target']]

    return DataFrame({
        'raw': raw,
        'label': labels
    })
