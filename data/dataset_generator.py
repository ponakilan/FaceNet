import os
import random

import pandas as pd


def get_files_list(root_dir):
    list_root_dir = os.listdir(root_dir)

    # Getting the complete path of the child directories
    child_dirs = []
    for child_dirs_t in list_root_dir:
        child_dirs.append(os.path.join(root_dir, child_dirs_t))

    # Creating a nested list containing all the files with complete paths
    files = []
    for directory in child_dirs:
        cur_files = []
        for file in os.listdir(directory):
            cur_files.append(os.path.join(directory, file))
        files.append(cur_files)

    return files


def generate_dataset(root_dir):
    files = get_files_list(root_dir)

    anchor_files, positive_files, negative_files = [], [], []
    while len(files) > 0:

        if len(files) == 2 and len(files[0]) == 0 or len(files[1]) == 0:
            break

        # Remove any empty lists in files
        while [] in files:
            files.remove([])
        
        # Select random folders for anchor_positive and negative
        anchor_pos_dir = random.choice(files)
        while len(anchor_pos_dir) < 2:
            anchor_pos_dir = random.choice(files)
        neg_dir = random.choice(files)
        while anchor_pos_dir == neg_dir:
            neg_dir = random.choice(files)
        
        # Select images for anchor, positive and negative
        anchor = random.choice(anchor_pos_dir)
        positive = random.choice(anchor_pos_dir)
        while anchor == positive:
            positive = random.choice(anchor_pos_dir)
        negative = random.choice(neg_dir)

        # Remove the selected files from the list
        anchor_pos_dir.remove(anchor)
        anchor_pos_dir.remove(positive)
        neg_dir.remove(negative)
        
        # Append all the files to the respective lists
        anchor_files.append(anchor)
        positive_files.append(positive)
        negative_files.append(negative)
    
    # Create a dataframe with the generated lists
    face_triplets_df = pd.DataFrame({
        "anchor": anchor_files,
        "positive": positive_files,
        "negative": negative_files
    })
    
    return face_triplets_df
