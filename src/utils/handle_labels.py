__author__ = 'timon'

import rasterio
import pyproj
import numpy as np
import const
import matplotlib.pyplot as plt

# The map projection used in the NLCD is Albers Conical Equal-Area (ACEA) coordinates (SR-ORG:6630)
# (http://spatialreference.org/ref/sr-org/albers-conical-equal-area-as-used-by-mrlcgov-nlcd/)
ACEA_PROJ4 = "+proj=aea +lat_1=29.5 +lat_2=45.5 +lat_0=23 +lon_0=-96" \
             "+x_0=0 +y_0=0 +ellps=GRS80 +datum=NAD83 +units=m +no_defs"

def get_labels_lat_lon(coords, nlcd):
    """Gets the labels corresponding to coordinates

    1. Transforms (latitude, longitude) to the coordinate reference system
       used in the NLCD data set (Alber Conical Equal Area (ACEA))
    2. Transforms ACEA coordinates to pixels in the raster data set
    3. Queries the labels for those pixels
       None if one of the indeces of the pixel is out-of-bounds

    :param coords: list of (latitude, longitude) tuples
    :param main_folder: path to folder where the data folder is found
    :return: list: list containing the labels corresponding to each coordinate
                   None for coordinates not in the NLCD data set
    """
    labels = []
    # transform lat, lon to Albers Conical Equal Area (ACEA)
    acea = pyproj.Proj(ACEA_PROJ4)
    acea_coords = [(acea(lon, lat)) for lat, lon in coords]
    # open NLCD raster data set
    with rasterio.drivers():
        with rasterio.open(nlcd) as src:
            # linear transformation between ACEA coordinates and pixels
            rev = ~src.affine
            # transform ACEA to pixel coordinates
            pixels = [tuple(int(round(i)) for i in rev*coord) for coord in acea_coords]
            for col, row in pixels:
                if col < 0 or col >= src.width or row < 0 or row >= src.height:
                    labels.append(None)
                else:
                    window = ((row, row+1), (col, col+1))
                    labels.append(src.read(1, window=window)[0,0])
    return labels

def get_labels_tif(geo_tiff, nlcd):
    """Returns all labels that correspond to the locations coverd by the GeoTiff

    :param geo_tiff: path to GeoTiff file
    :param nlcd: path to NLCD .img file
    :return: np.array: an array containing all the labels for the area covered by the GeoTiff
    """
    with rasterio.drivers():
        # open input file
        with rasterio.open(geo_tiff) as src:
            # get all coordinates of pixels
            coords = []
            width = src.width
            height = src.height
            for col in range(0, src.width):
                for row in range(0, src.height):
                    coords.append(src.affine*(col, row))
            # build Proj.4 string specifying CRS
            if 'init' not in src.crs:  # no CRS found or just not a standard one
                raise Exception("Invalid CRS! " + str(src.crs))
            proj4 = str('+init=' + src.crs['init'])
            proj = pyproj.Proj(proj4)

        # open NLCD
        with rasterio.open(nlcd) as src:
            # linear transformation between ACEA coordinates and pixels
            rev = ~src.affine
            # transform coordinates from the respective map projection to ACEA and from ACEA to pixels
            acea = pyproj.Proj(ACEA_PROJ4)
            coords = [rev*pyproj.transform(proj, acea, *coord) for coord in coords]
            pixels = list(tuple(map(lambda x: int(round(x)), coord)) for coord in coords)
            # get labels for pixels
            labels = []
            x_origin, y_origin = pixels[0]
            x_origin = x_origin - width
            y_origin = y_origin - height
            # speed up by reading all relevant labels from the NLCD at once
            window = ((y_origin, y_origin + 3*height), (x_origin, x_origin + 3*height))
            label_matrix = src.read(1, window=window)
            for col, row in pixels:
                label = label_matrix[(row - y_origin, col - x_origin)]
                # aggregate NLCD labels
                label = (label/10)*10
                labels.append(label)
        return labels

def visualize_labels(geo_tiff, nlcd):
    """Make a picture with color-coded labels as pixels

    :param geo_tiff: path to GeoTiff file
    :param nlcd: path to NLCD .img file
    """
    print "Getting labels..."
    labels = get_labels_tif(geo_tiff, nlcd)
    with rasterio.drivers():
        with rasterio.open(geo_tiff) as src:
            width = src.width
            height = src.height
    rgb = np.zeros((height, width, 3))

    print "Getting colors for labels..."
    for col in range(0, width):
        for row in range(0, height):
            label = labels[row + col*height]
            r, g, b = const.RGB_LABELS[label]
            rgb[(row, col, 0)] = r
            rgb[(row, col, 1)] = g
            rgb[(row, col, 2)] = b

    # show image
    print "Showing image now..."
    plt.imshow(rgb)
    plt.show()

