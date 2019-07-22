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

    dataset = load_files(container_path, random_state=0)

    label_names = dataset['target_names']
    raw = [idx.decode('utf-8', 'replace') for idx in dataset['data']]
    labels = [label_names[idx] for idx in dataset['target']]
    filenames = dataset['filenames']

    return DataFrame({
        'raw': raw,
        'label': labels,
        'path': filenames
    })
