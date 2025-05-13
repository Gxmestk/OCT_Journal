from boundary_extraction import *

def main() -> int:
    # Read CSV file
    nam = pd.read_csv("reference_csv/Namsonthi_file_path.csv")

    # Process each image
    for _, row in nam.iterrows():
            folder_path=row["Folder_path"],
            filename=row["File_name"].replace(".png","_ILM_morphology_wrapper.png")
            print(folder_path, filename)


if __name__ == "__main__":
    sys.exit(main())