import os
import random

import pandas as pd


def get_files_list(root_dir: str) -> list:
    """
    Generates a nested list of file paths for files inside the subfolders of a given directory,
    where each sublist corresponds to a single directory.

    Args:
        root_dir (str): Root directory containing subfolders with files.

    Returns:
        list: Nested list of file paths of the files present in the subfolders of the root directory.

    Example:
        # Getting the list of files from a root directory \n
        root_directory = "/path/to/root_directory" \n
        files_list = get_files_list(root_directory) \n
        print(files_list)\n

        Output:
        [['/path/to/root_directory/subfolder1/file1', '/path/to/root_directory/subfolder1/file2', ...], ...]
    """
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


def generate_dataset(root_dir: str) -> pd.DataFrame:
    """
    Generates triplets of images (anchor, positive, negative) from the images present in the given directory
    for training FaceNet with a triplet loss function.

    Args:
        root_dir (str): Root directory containing subfolders of images.

    Returns:
        pandas.DataFrame: Dataframe containing triplets of file paths (anchor, positive, negative).

    Example:
        # Generating a dataset of triplets from a root directory
        root_directory = "/path/to/root_directory" \n
        dataset = generate_dataset(root_directory) \n
        print(dataset.head())
    """
    files = get_files_list(root_dir)

    anchor_files, positive_files, negative_files = [], [], []
    while len(files) > 0:

        # Stop if there are only two folders and one of them is empty
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
