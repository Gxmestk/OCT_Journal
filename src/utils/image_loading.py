"""Module for loading and processing image dataset paths."""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from natsort import natsorted


def get_dataset_path_dict(dataset_path_dict: dict) -> tuple[pd.DataFrame, str | None]:
    """Get a dictionary mapping datasets to their folder paths and file names.

    Applies os.path.normpath() to all paths for cross-platform compatibility.

    Args:
    ----
        dataset_path_dict: Dictionary with dataset names as keys and lists of paths as values.

    Returns:
    -------
        tuple: (file_info DataFrame, first dataset name)
    """
    # Pre-allocate lists for better performance
    datasets = []
    folder_paths = []
    file_names_list = []

    for dataset, folders in dataset_path_dict.items():
        for folder_path in folders:
            # Normalize the path for cross-platform compatibility
            normalized_path = os.path.normpath(folder_path)

            # Verify path exists
            if not Path(normalized_path).exists():
                error_msg = f"Path not found: {normalized_path}"
                raise FileNotFoundError(error_msg)

            # Get and sort files
            try:
                files = natsorted(os.listdir(normalized_path))
            except OSError as e:
                error_msg = f"Error reading files from {normalized_path}: {e!s}"
                raise RuntimeError(error_msg) from e

            # Extend lists
            datasets.extend([dataset] * len(files))
            folder_paths.extend([normalized_path] * len(files))
            file_names_list.extend(files)

    # Create DataFrame in one operation
    file_info = pd.DataFrame(
        {
            "Dataset": datasets,
            "Folder_path": folder_paths,
            "File_name": file_names_list,
        },
    )

    return file_info, datasets[0] if datasets else None
