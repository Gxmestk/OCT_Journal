"""Process ILM images."""

from __future__ import annotations

import argparse
import sys
from typing import TYPE_CHECKING, TypedDict

import cv2
import pandas as pd

if TYPE_CHECKING:
    import numpy as np

from preprocessing.ilm_rpe import preprocessing_for_ilm


class ThresholdParams(TypedDict):
    """Type definition for threshold parameters."""

    thresh: float
    maxval: float
    type: int


def threshold_wrapper(img: np.ndarray, **kwargs: ThresholdParams) -> np.ndarray:
    """
    Apply thresholding to an image.

    Args:
    ----
        img: Input image
        **kwargs: Arguments to pass to cv2.threshold

    Returns:
    -------
        Thresholded image
    """
    _, thresh = cv2.threshold(img, **kwargs)
    return thresh


def morphology_wrapper(
    img: np.ndarray,
    operation: int,
    kernel_shape: tuple[int, int],
    kernel_type: int = cv2.MORPH_ELLIPSE,
    iterations: int = 1,
) -> np.ndarray:
    """
    Perform morphological operations on an image.

    Args:
    ----
        img: Input image
        operation: Morphological operation type
        kernel_shape: Size of the structuring element
        kernel_type: Shape of the structuring element (default: cv2.MORPH_ELLIPSE)
        iterations: Number of times operation is applied (default: 1)

    Returns:
    -------
        Processed image
    """
    kernel = cv2.getStructuringElement(kernel_type, kernel_shape)
    return cv2.morphologyEx(img, operation, kernel, iterations=iterations)




def main() -> int:
    """Preprocess ILM images."""
    # Define processing settings
    processing_settings = {
        cv2.GaussianBlur: {
            "ksize": (5, 5),
            "sigmaX": 0,
            "borderType": cv2.BORDER_REFLECT,
        },
        cv2.fastNlMeansDenoising: {
            "h": 9,
            "templateWindowSize": 15,
            "searchWindowSize": 35,
        },
        threshold_wrapper: {"thresh": 0, "maxval": 255, "type": cv2.THRESH_OTSU},
        morphology_wrapper: {
            "operation": cv2.MORPH_CLOSE,
            "kernel_shape": (5, 5),
            "kernel_type": cv2.MORPH_ELLIPSE,
            "iterations": 1,
        },
    }

    # Read CSV file
    dataset = pd.read_csv("reference_csv\\OCTID_file_path.csv")

    # Process each image
    dataset_name = str(dataset["Dataset"].iloc[0])
    
    for _, row in dataset.iterrows():
        preprocessing_for_ilm(
            folder_path=row["Folder_path"],
            filename=row["File_name"],
            output_folder=str(row["Folder_path"]).replace(dataset_name, f"PreProcessed_{dataset_name}"),
            processing_settings=processing_settings,
            save_image=True,
            save_metadata=False,
        )


    

    # Parse command line arguments if needed
    parser = argparse.ArgumentParser()
    parser.parse_args()
    return 0
















if __name__ == "__main__":
    sys.exit(main())
