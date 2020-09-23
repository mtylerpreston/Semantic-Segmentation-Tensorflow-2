import numpy as np
import os
from sklearn.model_selection import train_test_split


# train val test split
def train_val_test_split(path_to_data, split_probs={'train': .8, 'val': .2, 'test': .2}, random_state=99):
    '''
    path_to_data: string - path of directory for data. This should contain subdirectories 'images' and 'labels'.

    Creates train, val, and test directories as needed under path_to_data and randomly moves the images and labels into each split:
    path_to_data/train/images
    path_to_data/train/labels
    path_to_data/val/images
    path_to_data/val/labels
    path_to_data/test/images
    path_to_data/test/labels
    '''
    contents = os.listdir(path_to_data)
    if 'masks' in contents and 'labels' not in contents:
        os.rename(os.path.join(path_to_data, 'masks'), os.path.join(path_to_data, 'labels'))

    image_dir = os.path.join(path_to_data, 'images')
    label_dir = os.path.join(path_to_data, 'labels')

    image_names = os.listdir(image_dir)
    label_names = os.listdir(label_dir)

    train_images, test_images, train_labels, test_labels = train_test_split(image_names, label_names, test_size=split_probs['test'], random_state=random_state)
    train_images, val_images, train_labels, val_labels = train_test_split(train_images, train_labels, test_size=(split_probs['val']/(split_probs['train']+split_probs['val'])), random_state=random_state)
    
    train_path = os.path.join(path_to_data, 'train')
    val_path = os.path.join(path_to_data, 'val')
    test_path = os.path.join(path_to_data, 'test')

    if not os.path.isdir(train_path):
        os.mkdir(train_path)
        os.mkdir(os.path.join(train_path, 'images'))
        os.mkdir(os.path.join(train_path, 'labels'))
    if not os.path.isdir(val_path):
        os.mkdir(val_path)
        os.mkdir(os.path.join(val_path, 'images'))
        os.mkdir(os.path.join(val_path, 'labels'))
    if not os.path.isdir(test_path):
        os.mkdir(test_path)
        os.mkdir(os.path.join(test_path, 'images'))
        os.mkdir(os.path.join(test_path, 'labels'))

    for file_name in train_images:
        src_path = os.path.join(image_dir, file_name)
        dest_path = os.path.join(path_to_data, 'train', 'images', file_name)
        os.rename(src_path, dest_path)
    for file_name in train_labels:
        src_path = os.path.join(label_dir, file_name)
        dest_path = os.path.join(path_to_data, 'train', 'labels', file_name)
        os.rename(src_path, dest_path)
    for file_name in val_images:
        src_path = os.path.join(image_dir, file_name)
        dest_path = os.path.join(path_to_data, 'val', 'images', file_name)
        os.rename(src_path, dest_path)
    for file_name in val_labels:
        src_path = os.path.join(label_dir, file_name)
        dest_path = os.path.join(path_to_data, 'val', 'labels', file_name)
        os.rename(src_path, dest_path)
    for file_name in test_images:
        src_path = os.path.join(image_dir, file_name)
        dest_path = os.path.join(path_to_data, 'test', 'images', file_name)
        os.rename(src_path, dest_path)
    for file_name in test_labels:
        src_path = os.path.join(label_dir, file_name)
        dest_path = os.path.join(path_to_data, 'test', 'labels', file_name)
        os.rename(src_path, dest_path)

    os.rmdir(image_dir)
    os.rmdir(label_dir)

if __name__ == '__main__':
    train_val_test_split('datasets/endoscopy', split_probs={'train': .8, 'val': .19, 'test': .01}, random_state=99)
