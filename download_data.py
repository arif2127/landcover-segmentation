import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("adrianboguszewski/landcoverai")

print("Downloaded to:", path)

# Move dataset to project data/raw folder
destination = "data/raw/landcoverai"

if not os.path.exists(destination):
    shutil.copytree(path, destination)

print("Dataset moved to:", destination)