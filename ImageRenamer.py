from PIL import Image, ExifTags
import os

folders_to_fix = [f.path for f in os.scandir("/Volumes/DebrisImgWA/") if f.is_dir() and os.path.basename(f)[:4] == "2020"]

for folder_to_fix in folders_to_fix:
        print("Fixing filenames in folder ", folder_to_fix)
        files_in_folder = sorted([os.path.join(folder_to_fix, f) for f in os.listdir(folder_to_fix) if os.path.isfile(os.path.join(folder_to_fix, f)) and f[-4:].lower() == ".jpg" and f[0] != "_"])
        for i, old_filepath in enumerate(files_in_folder):
                old_filename = os.path.basename(old_filepath)
                if len(old_filename.split(' ')) == 1 and old_filename[0] != ".":  # checks for 2020 images that weren't already renamed with a sequence identifier
                        image = Image.open(old_filepath)
                        exif_data = image.getexif()
                        exif_tags = { ExifTags.TAGS[k]: v for k, v in exif_data.items() if k in ExifTags.TAGS and type(v) is not bytes }
                        datetime_with_underscores = exif_tags['DateTime'].replace(':', '_')
                        sequence_str = f'{i:05}'
                        new_filename = datetime_with_underscores + ' ' + sequence_str + '.jpg'
                        dirname = os.path.dirname(old_filepath)
                        # print("Renaming {0} to {1}.".format(old_filepath, os.path.join(dirname, new_filename)))
                        os.rename(old_filepath, os.path.join(dirname, new_filename))
