__author__ = 'timon'

import os
import glob
import json
import shutil
from collections import Counter

import planet_api
import tile_image
import handle_labels
import FeatureExtraction
import const

def download_pictures(original_dir):
    """download all the pictures we want"""
    # clean up
    os.system('rm -rf ' + original_dir + '/*')

    # gets all the images and saves them in "data/originals"
    planet_api.fetch_images(original_dir, scene_type='rapideye', product='analytic')

def make_tiles(original_dir, tiles_dir, nlcd):
    """make tiles out of the big pictures in data/originals"""

    # This threshold filters out tiles where the dominant label covers less than 90% of the pixels.
    # This ensures that the model only trains on tiles that exhibit one specific label, not i.e. half forest and
    # half desert.
    threshold = 0.9

    # clean up
    os.system('rm -rf ' + tiles_dir + '/*')

    count = 0

    for image in glob.glob(original_dir + '/*.tif'):
        # save count for reporting
        save_count = count
        # make temp directory for tiles from one image
        temp = tiles_dir + '/temp'
        if not os.path.isdir(temp):
            os.mkdir(temp)
        # split image into tiles and origin
        tile_image.tiles_from_image(image, temp)

        for tile in glob.glob(temp + '/*.tif'):
            if count%50 ==0:
                print "Fetching labels for tile number %d"%count

            # get labels for tile
            label_counter = Counter(handle_labels.get_labels_tif(tile, nlcd))
            # fix problem json has with non-string keys
            label_counter = {str(key):value for key, value in label_counter.items()}

            # only accept "pure" images to the data set
            num, _ =  max((label_counter[key], key) for key in label_counter)
            if num / (const.TILE_SIZE**2) < threshold:
                continue
            # fix weird thing in the NLCD data set
            # ignore all tiles that have 0 labels
            if u'0' in label_counter:
                continue
            # create new directory
            next_dir = os.path.join(tiles_dir, format(count, '08'))
            if not os.path.isdir(next_dir):
                os.mkdir(next_dir)
            # write to json

            with open(os.path.join(next_dir, format(count, '08') + '.json'), 'w') as outfile:
                json.dump(label_counter, outfile, sort_keys=True, indent=4)

            # move .tif, .txt to new directory
            os.system('mv ' + tile + ' ' + os.path.join(next_dir, format(count, '08') + '.tif'))
            os.system('mv ' + tile.replace('.tif', '.txt') + ' ' + os.path.join(next_dir, format(count, '08') + '.txt'))
            # increment counter for next tile directory
            count = count + 1

        print "Created %d tiles for image %s" % (count - save_count, image)
        # clean up
        os.system('rm -rf ' + temp + '/')

    print "Created %d tiles in total" % (count)

def extract_features(tiles_dir):
    # ectract all different features
    for requested_extractors in const.FEATURE_EXTS:
        feature_extractor = FeatureExtraction.JointFeatureExtractor(tiles_dir, requested_extractors)
        feature_extractor.generate_feature_files()

def split_data(train, validation, test):
    """split the dataset in 60% train 20% validation 20% test"""
    count = 0
    for tile_no in os.listdir(train):
        if count % 5 == 3:
            current_tile_dir = os.path.join(train, tile_no)
            new_tile_dir = os.path.join(validation, tile_no)
            shutil.move(current_tile_dir, new_tile_dir)
        if count % 5 == 4:
            current_tile_dir = os.path.join(train, tile_no)
            new_tile_dir = os.path.join(test, tile_no)
            shutil.move(current_tile_dir, new_tile_dir)
        count = count + 1
        if count % 100 == 0:
            print "Split %d tiles so far..." % count
