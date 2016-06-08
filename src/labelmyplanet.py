__author__ = 'timon'

import os
from PIL import Image
import utils.FeatureExtraction
import rasterio
import utils.const
import learning.Classifiers
import learning.training_preparation as tp
import utils.tile_image as ti
import glob
import optparse

def predict(geo_tiff, model_path):
    """
    Predict the land cover classes for the input image (at a resolution of 128x128 pixels)
    using a trained Random Forest

    1. Tile GeoTiff
    2. Make prediction for tiles
    3. Compute bounding boxes and colors for tiles (used in visualization)

    :param geo_tiff: path to GeoTiff file
    :param model_path: path to pickled model
    :return (bboxes, colors): the bounding boxes of the tiles and the colors (corresponding to land cover classes)
    """

    # Create temporary directory
    print "Creating temporary directories..."
    tiles_tmp = os.path.join(os.path.dirname(geo_tiff), "tiles_tmp")
    os.mkdir(tiles_tmp)

    # Tile image
    print "Tiling image..."
    ti.tiles_from_image(geo_tiff, tiles_tmp)

    # transformation from coordinates to pixels of big pic
    with rasterio.drivers():
        with rasterio.open(geo_tiff) as src:
            # linear transformation between coordinates (in the resp. CRS) and the pixels
            rev = ~src.affine

    count = 0
    bboxes = []
    print "Extracting features..."
    # Create Feature Extractor
    fe_ids = ['gab', 'ir_re_hist', 'hsv_hist']
    feature_extractor = utils.FeatureExtraction.JointFeatureExtractor('', requested_FE_ids=fe_ids)
    feature_type = feature_extractor.ending.strip('.')
    num_features = feature_extractor.feature_length

    list_of_tiles = glob.glob(tiles_tmp + '/*.tif')
    list_of_feature_files = []
    for tile in list_of_tiles:
        with rasterio.drivers():
            with rasterio.open(tile) as src:
                # linear transformation between pixels and coordinates (in the resp. CRS)
                fwd = src.affine
                top_left = rev*fwd*(0, 0)
                bottom_right = rev*fwd*(src.width, src.height)
                bboxes.append((int(top_left[0]), int(top_left[1]), int(bottom_right[0]), int(bottom_right[1])))

        # Extract Features
        list_of_feature_files.append(feature_extractor.generate_single_feature_file(tile))
        count = count + 1

    # load model
    model = learning.Classifiers.RandomForest_c()
    model.load(model_path)

    # read in data
    print "Reading in data..."
    X = tp.read_new_data(tiles_tmp, feature_type, num_features, list_of_feature_files)

    # predict new labels
    print "Predicting labels..."
    y_pred = model.predict(X)
    print "Labels found: " + str(set(y_pred))
    colors = []
    for label in y_pred:
        rgb_value = utils.const.RGB_LABELS[label]
        rgb_value =tuple(map(lambda x: int(round(100*x)), rgb_value))
        colors.append("rgb({0}%,{1}%,{2}%)".format(*rgb_value))

    # clean up
    remove_rec = "rm -rf " + tiles_tmp
    os.system(remove_rec)

    return (bboxes, colors)


def visualize(geo_tiff, bboxes, colors):
    """
    Visualize the predicted land cover classes

    1. Convert the original (5-band) GeoTiff to a color-corrected RGB
    2. Use the bounding box and color information to visualize the predicted land cover classes

    :param geo_tiff: path to GeoTiff file
    :param bboxes: list of bounding boxes for every tile
    :param colors: list of colors for every tile
    :return Nothing
    """
    # Temporary file paths
    dir = os.path.dirname(geo_tiff)
    rgb = os.path.join(dir, "rgb.tif")
    rgb_color = os.path.join(dir, "rgb_color.tif")
    rgb_bright = os.path.join(dir, "rgb_bright.tif")

    # Command Line to get a visual picture from RapidEye Imagery
    translate = "gdal_translate -b 3 -b 2 -b 1 " + geo_tiff + " " + rgb
    warp = "gdalwarp -co photometric=RGB " + rgb + " " + rgb_color
    convert = "convert -sigmoidal-contrast 30x8% -depth 8 " + rgb_color + " " + rgb_bright
    os.system(translate)
    os.system(warp)
    os.system(convert)

    image = Image.open(rgb_bright)

    # RGB version of the original image
    image.save(geo_tiff.strip('.tif') + '.jpeg', "JPEG")
    mask = Image.new("L", (128,128), 64)
    for i in range(len(colors)):
        image.paste(colors[i], bboxes[i], mask)

    # Original image overlayed with visualization of predicted land cover classes
    print "Saving visualization..."
    image.save(geo_tiff.strip('.tif') + '_pred.jpeg', "JPEG")

    # clean up
    remove = "rm -f " + rgb.strip('.tif') + "*"
    os.system(remove)

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-p', '--path', dest='path')
    parser.add_option('-d', '--directory', dest='directory')

    (options, _) = parser.parse_args()

    for geo_tiff in glob.glob(os.path.join(options.directory, '*.tif')):
        print geo_tiff
        bboxes, colors = predict(geo_tiff, options.path)
        visualize(geo_tiff, bboxes, colors)







