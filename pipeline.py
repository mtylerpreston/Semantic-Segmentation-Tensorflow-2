import numpy as np
import cv2
from PIL import Image
import glob
from sklearn.model_selection import train_test_split
import os


def train_val_test_split(path_to_data, split_probs={'train': .8, 'test': .2}, random_state=99):
    '''
    path_to_data: string - path of directory for data. This should contain subdirectories 'images' and 'labels'.

    Creates train, val, and test directories as needed under path_to_data and randomly moves the images and labels into each split:
    path_to_data/images/train
    path_to_data/images/test
    path_to_data/labels/train
    path_to_data/labels/test
    '''
    contents = os.listdir(path_to_data)
    if 'masks' in contents and 'labels' not in contents:
        os.rename(os.path.join(path_to_data, 'masks'), os.path.join(path_to_data, 'labels'))

    image_dir = os.path.join(path_to_data, 'images')
    label_dir = os.path.join(path_to_data, 'labels')

    image_names = os.listdir(image_dir)
    label_names = os.listdir(label_dir)

    train_images, test_images, train_labels, test_labels = train_test_split(image_names, label_names, test_size=split_probs['test'], random_state=random_state)
    
    if not os.path.isdir(os.path.join(path_to_data, 'images', 'train')):
        os.mkdir(os.path.join(path_to_data, 'images', 'train'))
    if not os.path.isdir(os.path.join(path_to_data, 'labels', 'train')):
        os.mkdir(os.path.join(path_to_data, 'labels', 'train'))
    if not os.path.isdir(os.path.join(path_to_data, 'images', 'test')):
        os.mkdir(os.path.join(path_to_data, 'images', 'test'))
    if not os.path.isdir(os.path.join(path_to_data, 'labels', 'test')):
        os.mkdir(os.path.join(path_to_data, 'labels', 'test'))
    
    for file_name in train_images:
        src_path = os.path.join(image_dir, file_name)
        dest_path = os.path.join(path_to_data, 'images', 'train', file_name)
        os.rename(src_path, dest_path)
    for file_name in train_labels:
        src_path = os.path.join(label_dir, file_name)
        dest_path = os.path.join(path_to_data, 'labels', 'train', file_name)
        os.rename(src_path, dest_path)
    for file_name in test_images:
        src_path = os.path.join(image_dir, file_name)
        dest_path = os.path.join(path_to_data, 'images', 'test', file_name)
        os.rename(src_path, dest_path)
    for file_name in test_labels:
        src_path = os.path.join(label_dir, file_name)
        dest_path = os.path.join(path_to_data, 'labels', 'test', file_name)
        os.rename(src_path, dest_path)

def clean_masks(path_to_data):
    files = glob.glob(os.path.join(path_to_data, '*', '*', '*'))
    labels = [f for f in files if 'labels' in f]
    for path in labels:
        label = cv2.imread(path, 0)
        label[label>125] = 255
        label[label<125] = 0
        new_path = path.split('.')[0]+'.png'
        cv2.imwrite(new_path, label)


if __name__ == '__main__':
    # train_val_test_split('datasets/endoscopy', split_probs={'train': .8, 'test': .2}, random_state=99)
    clean_masks('datasets/endoscopy')
