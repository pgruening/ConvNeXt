import os
from os.path import join, splitext, isfile
import shutil
import pandas as pd
import glob

# there should be no drawback in renaming the files!
# https://superuser.com/questions/367834/is-there-any-drawback-to-just-renaming-jpeg-files-to-jpg

#FOLDER_PATH = 'JPEG_vs_jpeg'
FOLDER_PATH = '/data/ILSVRC/Data'
CSV_PATH = '/data/LOC_val_solution.csv'
#CSV_PATH = 'JPEG_vs_jpeg/LOC_val_solution.csv'
VAL_DATA_PATH = '/data/ILSVRC/Data/CLS-LOC/val'


def rename_files():
    def is_JPEG(x):
        return splitext(x)[-1] == '.JPEG'

    for root, _, files_ in os.walk(FOLDER_PATH):
        files_ = [x for x in files_ if is_JPEG(x)]
        if not files_:
            continue

        for file in files_:
            new_name = f'{splitext(file)[0]}.jpeg'
            os.rename(join(root, file), join(root, new_name))


def move_val_images_to_class_folders():
    ctr = 0
    df = pd.read_csv(CSV_PATH)

    assert glob.glob(join(VAL_DATA_PATH, '*.jpeg'))

    for _, row in df.iterrows():
        im_name = f'{row["ImageId"]}.jpeg'
        image_path = join(VAL_DATA_PATH, im_name)
        if not isfile(image_path):
            continue
        label = row['PredictionString'].split(' ')[0]

        directory = join(VAL_DATA_PATH, label)
        if not os.path.exists(directory):
            os.makedirs(directory)

        new_path = join(directory, im_name)

        shutil.move(image_path, new_path)
        ctr += 1

    print(f'Moved {ctr} images')


if __name__ == '__main__':
    move_val_images_to_class_folders()
