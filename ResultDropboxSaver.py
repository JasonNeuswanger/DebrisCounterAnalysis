import os
import shutil

data_folders = [f.path for f in os.scandir("/Volumes/DebrisImgWA/") if f.is_dir() and os.path.basename(f)[:2] == "20"]
for data_folder in data_folders:
    results_folder = os.path.join(data_folder, "_results")
    if os.path.exists(results_folder):
        for file in os.scandir(results_folder):
            if "Size Class Counts.xlsx" in file.path:
                print(file.path)
                shutil.copy2(file.path, "/Users/Jason/Dropbox/SFR/Projects/2019 Chena Drift Project/Chena Drift Project Data/Debris Particle Counts/")