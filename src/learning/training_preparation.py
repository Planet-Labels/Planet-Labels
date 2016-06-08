__author__ = 'rishabh'

import os
import glob
import numpy as np
import json
import h5py
from collections import defaultdict

TRAIN_FRAC = 0.8

# Setting the random seed
np.random.seed(10)

def read_data(tiles_dir, feature_type, num_features):
    print 'Features read are of this type:', feature_type
    # Locate pth to images
    num_images = sum(1 for a in os.listdir(tiles_dir) if os.path.isdir(os.path.join(tiles_dir, a)))

    X = np.zeros((num_images, num_features))
    y = np.zeros((num_images))
    counter = 0

    if num_images == 0:
        raise Exception('No images were found. Check your data source or path. Have a good day!')

    print 'We found ', num_images, ' images. Reading their ' + feature_type + ' features now.'
    
    # Read in features and label for each image found
    for tile_no in os.listdir(tiles_dir):
        current_tile_dir = os.path.join(tiles_dir, tile_no)
        # features stored in file with extension FILE_ENDING
        current_feature_path = os.path.join(current_tile_dir, tile_no + '.' + feature_type)
        # labels stored in file ending with .json
        current_label_path = os.path.join(current_tile_dir, tile_no + '.json')

        if os.path.isdir(current_tile_dir):
            with open(current_label_path, 'r') as data_file:
                dic = json.load(data_file)
                label = max((dic[key], key) for key in dic)[1]
                y[counter] = label

            with h5py.File(current_feature_path, 'r') as data_file:
                X[counter,:] = data_file['features'][()]

            # Print out progress for the benefit of user
            counter += 1
            if counter %100 == 0:
                print 'Finished reading ' + str(counter) + ' images.'

    print "Read in %d images in total!" % counter
    X_result = X[0:counter, :]
    y_result = y[0:counter]

    return X_result, y_result

def read_new_data(tiles_dir, feature_type, num_features, list_of_features):
    print 'Features read are of this type:', feature_type
    # Locate pth to images
    num_images = len(glob.glob(tiles_dir + '/*.' + feature_type))

    X = np.zeros((num_images, num_features))
    counter = 0

    if num_images == 0:
        raise Exception('No images were found. Check your data source or path. Have a good day!')

    print 'We found ', num_images, ' images. Reading their ' + feature_type + ' features now.'

    # Read in features and label for each image found
    for tile in list_of_features:
        with h5py.File(tile, 'r') as data_file:
            X[counter,:] = data_file['features'][()]

        # Print out progress for the benefit of user
        counter += 1
        if counter %100 == 0:
            print 'Finished reading ' + str(counter) + ' images.'
    print "Read in %d images in total!" % counter

    return X

def split_data(X, y, frac=TRAIN_FRAC):
    # Get the number of training examples in total
    num_images = X.shape[0]

    # Permutation of the images to randomly divide into training and test data
    perm = np.random.permutation(range(num_images))

    # We have chosen frac of the data for training
    X_train = X[perm[:int(num_images*frac)], :]
    y_train = y[perm[:int(num_images*frac)]]
    X_test = X[perm[int(num_images*frac):],:]
    y_test = y[perm[int(num_images*frac):]]
    print 'Xtrain shape: ', X_train.shape
    print 'Xtest shape: ', X_test.shape
    print 'ytrain shape: ', y_train.shape
    print 'ytest shape ', y_test.shape
    return X_train, y_train, X_test, y_test
