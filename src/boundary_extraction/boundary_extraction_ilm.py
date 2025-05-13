from boundary_extraction import *
import pandas as pd
import cv2

import matplotlib.pyplot as plt
import matplotlib.image as mpimg



def main() -> int:
    # Read CSV file
    # nam = pd.read_csv("reference_csv/Namsonthi_file_path.csv")

    # # Process each image
    # for _, row in nam.iterrows():
    #         folder_path=row["Folder_path"],
    #         filename=row["File_name"].replace(".jpeg","_ILM_morphology_wrapper.png")
    #         print(folder_path, filename)

    binary_mask = cv2.imread("D:\\OCT_NM\\Test_folder\\NORMAL100_ILM_morphology_wrapper.png", cv2.IMREAD_GRAYSCALE)

    # Assuming 'binary_mask' is your input image
    x_coords, y_coords = extract_line(
        mask=binary_mask,
        thickness_threshold=20,  # Your x pixels thickness criteria
        search_range=30,          # Your search space parameter
        percent_start = 20
    )


    # Create figure
    plt.figure(figsize=(10, 6))
    plt.imshow(binary_mask)

    # Plot points
    plt.plot(x_coords, y_coords)

    plt.axis('off')  # Hide axes
    plt.show()


if __name__ == "__main__":
    sys.exit(main())