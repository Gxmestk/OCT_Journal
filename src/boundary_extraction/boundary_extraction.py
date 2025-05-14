"""Module for processing retinal images to extract the ILM layer."""

from __future__ import annotations

import json
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d


def interpolate_coordinates(
    old_x: list,
    old_y: list,
    new_x: list = None,
    kind: str = "linear",
    extrapolate: bool = True,
) -> (np.ndarray, np.ndarray):
    """
    Interpolate y-coordinates to new x-coordinates.

    Parameters:
        old_x (array-like): Original x-coordinates.
        old_y (array-like): Original y-coordinates (must be same length as old_x).
        new_x (array-like, optional): New x-coordinates to interpolate at.
                                     If None, uses `np.linspace(min, max, 100)`.
        kind (str): Interpolation method ('linear', 'nearest', 'cubic', etc.).
        extrapolate (bool): If True, allows extrapolation beyond original range.

    Returns:
        new_x (np.ndarray): New x-coordinates (same as input or auto-generated).
        new_y (np.ndarray): Interpolated y-coordinates at new_x.
    """
    # Convert inputs to numpy arrays
    old_x = np.array(old_x)
    old_y = np.array(old_y)

    # Set up interpolation function
    if extrapolate:
        interp_func = interp1d(old_x, old_y, kind=kind, fill_value="extrapolate")
    else:
        interp_func = interp1d(
            old_x, old_y, kind=kind, bounds_error=False, fill_value=np.nan
        )

    # Compute interpolated y-values
    new_y = interp_func(new_x)
    new_y = np.round(new_y).astype(int)

    return new_x, new_y


def generate_search_offsets(search_range: int) -> int:
    """Generate a list of search offsets around a reference point."""
    offsets = [0]
    for i in range(1, search_range):
        offsets.extend([i, -i])
    return offsets


def find_reference_column(
    mask: np.ndarray,
    line: str,
    start_x: int,
    end_x: int,
    step: int,
    thickness_threshold: int,
) -> (int, int):
    """
    Find a reference column that meets the thickness criteria.
    Returns the x position and the first y position of the valid white cluster.
    """
    if line == "ILM":
        for x in range(start_x, end_x, step):
            for y in range(mask.shape[0]):
                if mask[y, x] > 0:  # Found a white pixel
                    # Check if there's a thick enough white cluster
                    if y + thickness_threshold <= mask.shape[0] and np.all(
                        mask[y : y + thickness_threshold, x] == 255
                    ):
                        return x, y
                    else:
                        # Skip past this white area
                        while y < mask.shape[0] and mask[y, x] > 0:
                            y += 1
    elif line == "RPE":
        for x in range(start_x, end_x, step):
            for y in range(mask.shape[0] - 1, -1, -1):
                if mask[y, x] > 0:  # Found a white pixel
                    # Check if there's a thick enough white cluster
                    if y - thickness_threshold >= 0 and np.all(
                        mask[y : y - thickness_threshold, x] == 255
                    ):
                        return x, y
                    else:
                        # Skip past this white area
                        while y < 0 and mask[y, x] > 0:
                            y -= 1
    return x, y


def trace_line_from_reference(
    mask: np.ndarray,
    line: str,
    start_x: int,
    end_x: int,
    step: int,
    initial_y: int,
    search_range: int,
) -> (np.ndarray, np.ndarray):
    """
    Trace the top surface line from a reference point.
    Returns lists of x and y coordinates.
    """
    x_vals = []
    y_vals = []
    y = initial_y
    search_offsets = generate_search_offsets(search_range)

    for x in range(start_x, end_x, step):
        if not np.any(mask[:, x] > 0):  # No white pixels in this column
            if np.any(mask[:, x:] > 0):  # Check if there are white pixels ahead
                continue
            else:
                break  # End of segment

        if line == "ILM":
            for offset in search_offsets:
                current_y = y + offset
                # Check boundaries and top surface condition
                if (
                    0 <= current_y < mask.shape[0]
                    and mask[current_y, x] > 0
                    and (current_y == 0 or mask[current_y - 1, x] == 0)
                ):
                    x_vals.append(x)
                    y_vals.append(current_y)
                    y = current_y
                    break

        elif line == "RPE":
            for offset in search_offsets:
                current_y = y + offset
                # Check boundaries and bottom surface condition
                if (
                    0 <= current_y < mask.shape[0]
                    and mask[current_y, x] > 0
                    and (current_y == 0 or mask[current_y + 1, x] == 0)
                ):
                    x_vals.append(x)
                    y_vals.append(current_y)
                    y = current_y
                    break

        # if not found:
        #     # x_vals.append(x)
        #     # y_vals.append(y)

    return x_vals, y_vals


def extract_line(mask, line, thickness_threshold, search_range, percent_start):
    """
    Main function to extract the top surface line from a binary mask.
    Returns x and y coordinates of the line.
    """

    # First try to find reference point in left segment (from 35% to start)
    ref_x, ref_y = find_reference_column(
        mask,
        line=line,
        start_x=int(mask.shape[1] * percent_start / 100),
        end_x=-1,
        step=-1,
        thickness_threshold=thickness_threshold,
    )

    # If not found in left segment, try right segment (from 35% to end)
    if ref_x is None:
        ref_x, ref_y = find_reference_column(
            mask,
            line=line,
            start_x=int(mask.shape[1] * percent_start / 100),
            end_x=mask.shape[1],
            step=1,
            thickness_threshold=thickness_threshold,
        )

    # If still not found, raise error
    if ref_x is None:
        raise ValueError(
            "No valid reference column found in either left or right segment"
        )

    # Trace left segment (from ref_x to start)
    x_vals_left, y_vals_left = trace_line_from_reference(
        mask,
        line=line,
        start_x=ref_x,
        end_x=-1,
        step=-1,
        initial_y=ref_y,
        search_range=search_range,
    )

    # Reverse left segment to get correct order
    x_vals_left = x_vals_left[::-1]
    y_vals_left = y_vals_left[::-1]

    # Trace right segment (from ref_x + 1 to end)
    x_vals_right, y_vals_right = trace_line_from_reference(
        mask,
        line=line,
        start_x=ref_x + 1,
        end_x=mask.shape[1],
        step=1,
        initial_y=y_vals_left[-1],
        search_range=search_range,
    )

    # Combine left and right segments
    x_vals = x_vals_left + x_vals_right
    y_vals = y_vals_left + y_vals_right

    if len(x_vals) != x_vals[-1] - x_vals[0] + 1:
        new_x = np.arange(x_vals[0], x_vals[-1] + 1)
        x_vals, y_vals = interpolate_coordinates(
            x_vals, y_vals, new_x=new_x, kind="linear", extrapolate=True
        )

    return x_vals, y_vals


def extract_ilm():

    # Read CSV file
    dataset = pd.read_csv("reference_csv\\OCTID_file_path.csv")

    # Process each image
    dataset_name = str(dataset["Dataset"].iloc[0])

    # Process each image
    for _, row in dataset.iterrows():
        
        preprocessed_folder = str(row["Folder_path"]).replace(dataset_name, f"PreProcessed_{dataset_name}")
        filename = row["File_name"]

        # Strip out the file extension
        basename, file_ext = os.path.splitext(filename)
        print(f"{preprocessed_folder}/{basename}_ILM_morphology_wrapper.png")

    binary_mask = cv2.imread("data/PreProcessed_OCTID/AMD/AMRD1_ILM_morphology_wrapper.png", cv2.IMREAD_GRAYSCALE)
    # Create figure with grayscale colormap
    plt.figure(figsize=(10, 6))

    # Display image in grayscale
    plt.imshow(binary_mask, cmap="gray")
    # Show the plot
    plt.show()
    

    #     # Assuming 'binary_mask' is your input image
    #     x_coords, y_coords = extract_line(
    #         mask=binary_mask,
    #         line="ILM",
    #         thickness_threshold=20,  # Your x pixels thickness criteria
    #         search_range=30,  # Your search space parameter
    #         percent_start=20,
    #         )



    #     # Create figure with grayscale colormap
    #     plt.figure(figsize=(10, 6))

    #     # Display image in grayscale
    #     plt.imshow(binary_mask, cmap="gray")

    #     # Plot points with red line (using 'r-' for red solid line)
    #     plt.plot(x_coords, y_coords, "r-", linewidth=2)  # 'r-' = red line

    #     plt.axis("off")  # Hide axes

    #     return 
    # # Save the figure before showing it
    # save_path = "D:\\OCTID_NM\\Test_Folder\\NORMAL100_ILM_boundary.png"  # Change this to your desired path
    # plt.savefig(save_path, bbox_inches="tight", pad_inches=0, dpi=300)

    # ILM_json = {"ILM_y_list": ILM_y_arr, "ILM_x_list": ILM_x_arr}

    # with open(
    #     "/content/drive/MyDrive/OCT_conference/[Test_AB]ILM_boundary_list2.json", "w"
    # ) as f:
    #     # indent=2 is not needed but makes the file human-readable
    #     # if the data is nested
    #     json.dump(ILM_json, f, indent=2)


def extract_rpe():
    binary_mask = cv2.imread(
        "D:\\OCTID_NM\\Test_Folder\\NORMAL100_ILM_threshold_wrapper.png",
        cv2.IMREAD_GRAYSCALE,
    )

    # Assuming 'binary_mask' is your input image
    x_coords, y_coords = extract_line(
        mask=binary_mask,
        line="RPE",
        thickness_threshold=20,  # Your x pixels thickness criteria
        search_range=30,  # Your search space parameter
        percent_start=20,
    )

    # Create figure
    plt.figure(figsize=(10, 6))
    plt.imshow(binary_mask)

    # Plot points
    plt.plot(x_coords, y_coords)

    plt.axis("off")  # Hide axes
    plt.show()
