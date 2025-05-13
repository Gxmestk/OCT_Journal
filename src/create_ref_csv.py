"""Create reference CSV files from dataset configurations."""

from __future__ import annotations

import argparse
import json
import sys
import re
from natsort import natsorted
from pathlib import Path
from typing import Any, Dict, List, Tuple
import pandas as pd


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process dataset paths and save information",
    )
    parser.add_argument(
        "--config",
        help="Path to dataset configuration JSON file",
        default="dataset_config.json",
    )
    parser.add_argument(
        "--save_path",
        help="Directory path to save the output CSV file",
        required=True,
    )
    return parser.parse_args()


def load_dataset_config(config_path: str) -> Dict[str, Any]:
    """Load dataset paths from JSON configuration file."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with config_file.open() as f:
        config: Dict[str, Any] = json.load(f)

    # Convert all paths to POSIX format
    converted_config: Dict[str, Any] = {}
    for key, paths in config.items():
        if isinstance(paths, list):
            converted_config[key] = [Path(p).as_posix() for p in paths]
        else:
            converted_config[key] = Path(paths).as_posix()

    return converted_config


def is_image_file(filename: str) -> bool:
    """Check if a file is an image based on its extension."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    return Path(filename).suffix.lower() in image_extensions


def extract_file_info(file_name: str) -> Tuple[str, str]:
    """Extract pre_screen and category from file name."""
    # Remove file extension and numbers
    base_name = Path(file_name).stem
    clean_name = re.sub(r'\d+', '', base_name).lower()
    
    if clean_name.startswith('abnormal'):
        return ('Abnormal', 'Abnormal')
    elif clean_name.startswith('normal'):
        return ('Normal', 'Normal')
    else:
        # For other cases, use the first word as category
        category = re.split(r'[_\-\s]', clean_name)[0]
        if not category:  # Fallback if no valid category found
            category = 'unknown'
        return ('Abnormal', category)


def get_file_paths(dataset_path_dict: Dict[str, Any]) -> List[Tuple[str, str, str, str, str]]:
    """Get all image file paths with additional information."""
    file_entries = []
    
    for dataset_name, paths in dataset_path_dict.items():
        paths_list = [paths] if isinstance(paths, str) else paths
        
        for path in paths_list:
            path_obj = Path(path)
            if not path_obj.exists():
                continue
                
            for item in path_obj.rglob('*'):
                # Skip hidden directories and files
                if any(part.startswith('.') for part in item.parts):
                    continue
                if item.is_file() and is_image_file(item.name):
                    folder_path = item.parent.as_posix()
                    file_name = item.name
                    pre_screen, category = extract_file_info(file_name)
                    file_entries.append((
                        dataset_name,
                        pre_screen,
                        category,
                        folder_path,
                        file_name
                    ))
    
    return file_entries


def main() -> int:
    """Process dataset paths and save to CSV."""
    args = parse_args()

    try:
        save_path = Path(args.save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Load configuration
        dataset_path_dict = load_dataset_config(args.config)

        # Get the dataset name from the config (first key in the dictionary)
        dataset_name = next(iter(dataset_path_dict.keys())) if dataset_path_dict else "dataset"
        
        # Get all file paths with additional info
        file_entries = get_file_paths(dataset_path_dict)

        # Create DataFrame with all columns and sort naturally
        df = pd.DataFrame(
            file_entries,
            columns=["Dataset", "Pre_Screen", "Category", "Folder_path", "File_name"]
        )
        df = df.iloc[natsorted(df.index, key=lambda x: df.loc[x, 'File_name'])]

        # Use dataset name from config for CSV filename
        output_file = save_path / f"{dataset_name}_file_path.csv"
        df.to_csv(output_file, index=False)
        print(f"Successfully saved to {output_file}")

    except FileNotFoundError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Error parsing JSON: {e}\n")
        return 1
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())