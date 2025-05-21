"""Module for processing retinal images to extract the ILM layer."""

from __future__ import annotations

import contextlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import cv2

if TYPE_CHECKING:
    import numpy as np

# Constants
DIM = 2


@dataclass
class ImageMetadata:
    """Container for image metadata parameters."""

    width: int
    height: int
    channels: int


class ProcessingParameters(TypedDict, total=False):
    """Type definition for processing parameters dictionary."""

    parameters: dict[str, int | float | str]
    info: dict[str, int | float | str]


def save_processed_image(
    image: np.ndarray,
    layer: str,
    process_name: str,
    filename: str,
    output_folder: str | Path,
) -> str:
    """Save processed image with standardized naming convention."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    base_name = Path(filename).stem
    output_image_name = f"{base_name}_{layer}_{process_name}.png"
    output_file = output_path / output_image_name
    cv2.imwrite(str(output_file), image)
    return str(output_file)


def save_process_metadata(
    image: np.ndarray,
    layer: str,
    process_name: str,
    filename: str,
    output_folder: str | Path,
    additional_metadata: ProcessingParameters | None = None,
) -> str:
    """
    Save metadata for processing step with standardized naming convention.

    Args:
    ----
        image: Processed image array
        layer: Layer identifier
        process_name: Name of the processing step
        filename: Original filename
        output_folder: Directory to save metadata
        additional_metadata: Extra metadata to include (default: None)

    Returns:
    -------
        Path to saved metadata file
    """
    additional_metadata = additional_metadata or {}

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    base_name = Path(filename).stem
    output_metadata_name = f"{base_name}_{layer}_{process_name}.json"
    output_file = output_path / output_metadata_name

    metadata = {
        "original_filename": filename,
        "processing_step": process_name,
        "timestamp": datetime.now(UTC).isoformat(),
        "image_properties": ImageMetadata(
            width=image.shape[1],
            height=image.shape[0],
            channels=1 if len(image.shape) == DIM else image.shape[DIM],
        ).__dict__,
        "processing_parameters": additional_metadata.get("parameters", {}),
        "additional_info": additional_metadata.get("info", {}),
    }

    with output_file.open("w") as f:
        json.dump(metadata, f, indent=4)
    return str(output_file)


class ImageProcessingError(Exception):
    """Custom exception for image processing errors."""


def _process_single_operation(
    image: np.ndarray,
    processing_func: Callable[..., np.ndarray],
    params: dict[str, int | float | str]
    | list[int | float | str]
    | tuple[int | float | str, ...],
) -> np.ndarray:
    """Process a single image operation with error handling."""
    if isinstance(params, (list, tuple)):
        return processing_func(image, *params)
    if isinstance(params, dict):
        return processing_func(image, **params)
    msg = "Parameters must be list/tuple (positional) or dict (keyword)"
    raise TypeError(msg)


def process_image_with_settings(
    layer: str,
    filename: str,
    output_folder: str | Path,
    image: np.ndarray,
    settings: dict[
        Callable[..., np.ndarray],
        dict[str, int | float | str]
        | list[int | float | str]
        | tuple[int | float | str, ...],
    ],
    *,
    save_image: bool = False,
    save_metadata: bool = False,
) -> np.ndarray:
    """
    Process an image using a sequence of operations defined in a settings dictionary.

    Args:
    ----
        layer: Layer identifier
        filename: Original filename
        output_folder: Directory for output files
        image: Input image to process
        settings: Dictionary mapping processing functions to their parameters
        save_image: Whether to save intermediate images (default: False)
        save_metadata: Whether to save processing metadata (default: False)

    Returns:
    -------
        Processed image after applying all operations

    Raises:
    ------
        ImageProcessingError: If an error occurs during processing
    """
    current_image = image.copy()
    errors = []

    for processing_func, params in settings.items():
        with contextlib.suppress(Exception) as exc:
            current_image = _process_single_operation(
                current_image,
                processing_func,
                params,
            )

            if save_image:
                save_processed_image(
                    current_image,
                    layer,
                    processing_func.__name__,
                    filename,
                    output_folder,
                )
            if save_metadata:
                save_process_metadata(
                    current_image,
                    layer,
                    processing_func.__name__,
                    filename,
                    output_folder,
                    {"parameters": params, "info": {}},
                )

        if exc is not None:
            errors.append(f"{processing_func.__name__}: {exc!s}")

    if errors:
        msg = f"Processing errors occurred:\n{'\n'.join(errors)}"
        raise ImageProcessingError(msg)

    return current_image



def fill_ILM(image: np.ndarray,
            output_folder: str | Path,
            filename:str,
            x_vals: list,
            y_vals: list) -> np.ndarray:
    
    for i in range(len(x_vals)):
        image[:y_vals[i] , x_vals[i]] = 0


    if save_image:
        save_processed_image(
            image,
            layer,
            fill_ILM.__name__,
            filename,
            output_folder,
        )
    if save_metadata:
        save_process_metadata(
            current_image,
            layer,
            fill_ILM.__name__,
            filename,
            output_folder,
            {"parameters": params, "info": {}},
        )

    return image


def read_json_boundary( file_path: str, 
                        line: str,
                        dataset_name: str,
                        image_basename: str):
    
    # Read the JSON file
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    if image_basename in (data[f"{line}_x_list_{dataset_name}"] and data[f"{line}_y_list_{dataset_name}"]):
        return data[f"{line}_x_list_{dataset_name}"][image_basename], data[f"{line}_y_list_{dataset_name}"][image_basename]


def preprocess_rpe():


    # Read CSV file
    dataset = pd.read_csv("reference_csv\\OCTID_file_path.csv")

    # Process each image
    dataset_name = str(dataset["Dataset"].iloc[0])

    return 


def preprocessing_for_ilm(
    folder_path: str | Path,
    filename: str,
    output_folder: str | Path,
    processing_settings: dict[
        Callable[..., np.ndarray],
        dict[str, int | float | str]
        | list[int | float | str]
        | tuple[int | float | str, ...],
    ],
    *,
    save_image: bool = False,
    save_metadata: bool = False,
) -> None:
    """
    Perform preprocessing pipeline for ILM layer extraction.

    Args:
    ----
        folder_path: Directory containing input image
        filename: Name of input image file
        output_folder: Directory to save processed outputs
        processing_settings: Dictionary of processing functions and their parameters
        save_image: Whether to save intermediate images (default: False)
        save_metadata: Whether to save processing metadata (default: False)
    """
    image_path = str(Path(folder_path) / filename)
    image_path = image_path.replace("\\", "/")
    # print(image_path)
    original_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if original_image is None:
        msg = f"Could not read image: {image_path}"
        raise ValueError(msg)

    with contextlib.suppress(ImageProcessingError):
        process_image_with_settings(
            layer="ILM",
            filename=filename,
            output_folder=output_folder,
            image=original_image,
            settings=processing_settings,
            save_image=save_image,
            save_metadata=save_metadata,
        )


def preprocessing_for_rpe(
    folder_path: str | Path,
    filename: str,
    output_folder: str | Path,
    ilm_bound_path : str | Path,
    processing_settings: dict[
        Callable[..., np.ndarray],
        dict[str, int | float | str]
        | list[int | float | str]
        | tuple[int | float | str, ...],
    ],
    *,
    save_image: bool = False,
    save_metadata: bool = False,
) -> None:
    """
    Perform preprocessing pipeline for ILM layer extraction.

    Args:
    ----
        folder_path: Directory containing input image
        filename: Name of input image file
        output_folder: Directory to save processed outputs
        processing_settings: Dictionary of processing functions and their parameters
        save_image: Whether to save intermediate images (default: False)
        save_metadata: Whether to save processing metadata (default: False)
    """
    image_path = str(Path(folder_path) / filename)
    image_path = image_path.replace("\\", "/")
    original_image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)

    if original_image is None:
        msg = f"Could not read image: {image_path}"
        raise ValueError(msg)

    # Strip out the file extension
    basename, file_ext = os.path.splitext(filename)

    # Read CSV file
    dataset = pd.read_csv("reference_csv\\OCTID_file_path.csv")

    # Process each image
    dataset_name = str(dataset["Dataset"].iloc[0])

    x_coords, y_coords = read_json_boundary(file_path = ilm_bound_path, 
                                            line = "ILM",
                                            dataset_name = dataset_name,
                                            image_basename = basename)

    filled_image = fill_ILM(image = original_image,
                            filename = basename,
                            x_vals = x_coords,
                            y_vals = y_coords,
                            save_image=save_image,
                            save_metadata=save_metadata)
    



    with contextlib.suppress(ImageProcessingError):
        process_image_with_settings(
            layer="RPE",
            filename=filename,
            output_folder=output_folder,
            image=filled_image,
            settings=processing_settings,
            save_image=save_image,
            save_metadata=save_metadata,
        )
