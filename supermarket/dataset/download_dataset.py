import os
import importlib
import urllib.parse
import zipfile
import subprocess
spec = importlib.util.find_spec("py7zr")
if spec is None:
    print("py7zr library not found. Installing it now...")
    subprocess.check_call(['pip', 'install', 'py7zr'])
import py7zr

def install_gdown():
    subprocess.check_call(['pip', 'install', 'gdown'])
def extract_zip_file(zip_file_path, destination_folder):
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)

def extract_7z_file(archive_file_path, destination_folder):
    with py7zr.SevenZipFile(archive_file_path, 'r') as archive_ref:
        archive_ref.extractall(destination_folder)
def download_file_from_google_drive(file_id, destination_folder):
    # Create the destination folder if it doesn't exist
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    # Check if gdown is installed, and install it if necessary
    spec = importlib.util.find_spec("gdown")
    if spec is None:
        print("gdown library not found. Installing it now...")
        install_gdown()
    import gdown
    # Use gdown to download the file
    url = f"https://drive.google.com/uc?id={file_id}"
    output = os.path.join(destination_folder, "indoor_trajectory_forecasting_dataset.zip")  # Replace .ext with the actual file extension
    gdown.download(url, output, quiet=False)
    if not os.path.exists(destination_folder+"/german"):
        os.makedirs(destination_folder+"/german")
    # Extract contents from the downloaded file
    extract_zip_file("indoor_trajectory_forecasting_dataset.zip", destination_folder+"/german")

    # Find .7z files and extract their contents
    for root, dirs, files in os.walk(destination_folder+"/german"):
        for file in files:
            if file.endswith(".7z"):
                archive_file_path = os.path.join(root, file)
                extract_7z_file(archive_file_path, root)

# Example usage
file_id = "10aIN5peOzb-zNjtnRXodo4mbuz3FNuZe"
destination_folder = "./"
download_file_from_google_drive(file_id, destination_folder)
# 10aIN5peOzb-zNjtnRXodo4mbuz3FNuZe

# https://drive.google.com/file/d/10aIN5peOzb-zNjtnRXodo4mbuz3FNuZe/view?usp=sharing