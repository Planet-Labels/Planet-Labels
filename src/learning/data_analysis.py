__author__ = 'rishabh'

import os
import numpy as np
import json
import matplotlib.pyplot as plt
import rasterio
import pyproj
import collections

def write_label_histogram(tiles_dir, file_path, landcover = [], balance=0):
    """Write histogram of labels found in tiles_dir to file_path"""
    # Locate pth to images
    num_images = sum(1 for a in os.listdir(tiles_dir) if os.path.isdir(os.path.join(tiles_dir, a)))

    if num_images == 0:
        raise Exception('No images were found. Check your data source or path. Have a good day!')

    print 'We found ', num_images, ' images. Reading their labels now'

    labs = collections.Counter()
    counter = 0

    for tile_no in os.listdir(tiles_dir):
        current_tile_dir = os.path.join(tiles_dir, tile_no)
        # labels stored in file ending with .json
        current_label_path = os.path.join(current_tile_dir, tile_no + '.json')

        if os.path.isdir(current_tile_dir):
            with open(current_label_path, 'r') as data_file:
                dic = json.load(data_file)
                num, label =  max((dic[key], key) for key in dic)
                # check if only certain landcover types should be read in
                if landcover:
                    if label not in landcover:
                        continue
                # only read in a up to balance of each label
                if balance > 0:
                    if labs[label] > balance:
                        continue
                labs[label] += 1

        counter += 1
        if counter %100 == 0:
            print 'Finished reading ' + str(counter) + ' images.'

    print "Found %s tiles" % sum(labs.values())
    print "Writing label histogram to file..."
    with open(file_path, "w") as file:
        json.dump(labs, file, sort_keys=True, indent=4)

def plot_label_histogram(file_path):
    with open(file_path, 'r') as data_file:
        label_dict = json.load(data_file)
    vals = []
    ticks = []
    dic = {10: "Water", 20:"Developed", 30:"Barren",
           40:"Forest", 50:"Shrubland", 70:"Herbaceous",
           80:"Cultivated", 90:"Wetlands"}
    for l in label_dict:
        vals.append(label_dict[l])
        ticks.append(dic[int(l)])
    bars = np.arange(len(vals))

    print 'Plotting histogram!'
    plt.style.use('ggplot')
    plt.figure()
    plt.bar(bars, vals, align="center")
    plt.xticks(bars, ticks, rotation=90)
    plt.show()

def get_lat_lon(geo_tiff):
    """Return latitude and longitude of center of geo_tiff"""
    with rasterio.drivers():
        with rasterio.open(geo_tiff) as src:
            # linear transformation between pixels and coordinates (in the resp. CRS)
            fwd = src.affine
            x, y = fwd*(src.width, src.height)
            # get projection for CRS
            if 'init' not in src.crs:  # no CRS found or just not a standard one
                raise Exception("Invalid CRS! " + str(src.crs))
            proj4 = str('+init=' + src.crs['init'])
            proj = pyproj.Proj(proj4)
            lon, lat = proj(x, y, inverse=True)
    return lat, lon

if __name__ == '__main__':
    main_folder = "/Volumes/500GB/Data/planet-labs/"
    tiles_dir = os.path.join(main_folder, 'data', 'tiles')
    file = "/Users/timon/Desktop/label_histogram.hdf5"
    write_label_histogram(tiles_dir, file)
    plot_label_histogram(file)

