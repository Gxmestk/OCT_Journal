"""Create reference CSV files from dataset configurations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from utils.image_loading import get_dataset_path_dict


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns
    -------
        Parsed command line arguments.
    """
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


def load_dataset_config(config_path: str) -> dict[str, str | list[str]]:
    """Load dataset paths from JSON configuration file and convert paths to OS-native format.

    Args:
    ----
        config_path: Path to the JSON configuration file.

    Returns:
    -------
        Dictionary containing the converted configuration paths.

    Raises:
    ------
        FileNotFoundError: If the config file doesn't exist.
    """
    config_file = Path(config_path)
    if not config_file.exists():
        msg = f"Config file not found at {config_path}"
        raise FileNotFoundError(msg)

    with config_file.open() as f:
        config: dict[str, Any] = json.load(f)

    # Convert all paths in the config to native OS format
    converted_config: dict[str, str | list[str]] = {}
    for key, paths in config.items():
        if isinstance(paths, list):
            converted_config[key] = [Path(p).as_posix() for p in paths]
        else:
            converted_config[key] = Path(paths).as_posix()

    return converted_config


def main() -> int:
    """Process dataset paths and save to CSV.

    Returns
    -------
        Return code (0 for success, 1 for failure).
    """
    args = parse_args()

    try:
        save_path = Path(args.save_path)

        # Create save directory if it doesn't exist
        save_path.mkdir(parents=True, exist_ok=True)

        # Load and convert dataset configuration
        dataset_path_dict = load_dataset_config(args.config)

        # Process the dataset paths
        df, dataset = get_dataset_path_dict(dataset_path_dict)

        # Save the dataframe to CSV
        save_file = save_path / f"{dataset}_file_path.csv"
        df["Folder_path"] = df["Folder_path"].str.replace("\\", "/")
        df.to_csv(save_file, index=False)

    except FileNotFoundError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 1
    except json.JSONDecodeError as e:
        sys.stderr.write(f"Error parsing JSON: {e}\n")
        return 1
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"Unexpected error: {e}\n")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
