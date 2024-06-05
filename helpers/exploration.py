import os
import pandas as pd

def traverse_dirs(dir_list: str) -> pd.DataFrame:

    rows = []
    for dir_path in dir_list:
        for dirname, _, filenames in os.walk(dir_path):
            if dirname in dir_list:
                continue
            if dirname.split("/")[-1] in ["train", "test", "holdout"]:
                split_type = dirname.split("/")[-1]
                continue
            if (len(filenames) > 0):
                class_name = dirname.split("/")[-1]
                n_images = len(filenames)
                row = [dir_path, split_type, class_name, n_images]
                rows.append(row)
    df = pd.DataFrame(rows, columns=['dataset', 'split_type', 'class', 'image_count'])
    return df