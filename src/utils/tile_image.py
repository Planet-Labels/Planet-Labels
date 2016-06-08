__author__ = 'vincent'

import os
import rasterio
import const
import optparse

def tiles_from_image(image_file_path, tiles_dir_path):
    """
    Splits one geotiff into tiles of size tile_size * tile_size that are then saved with correct "affine" information
    and "height" and "width" metadata.

    :param tile_size: The edge sizes of the tiles
    :param image_file_path: The file path to the geotiff that should be split in tiles.
    :param tiles_dir_path: The path to the directory where the tiles should be saved.
    :return: Nothing.
    """

    with rasterio.drivers():
        # open the image with rasterio
        with rasterio.open(image_file_path) as src:
            # read in the satelite image
            imarray = src.read()

            # Since the image will be split up into 256 * 256 tiles, we throw away columns and rows such that
            # the width and height are divisible by 256. These columns and rows would have been thrown away anyway,
            # since they contain whitespace.
            width_mod = src.width % const.TILE_SIZE
            height_mod = src.height % const.TILE_SIZE

            imarray = imarray[:, :-height_mod, :-width_mod]

            # calculate the number of tiles
            col_count = int(1.0 * src.width/ const.TILE_SIZE)
            row_count = int(1.0 * src.height / const.TILE_SIZE)
            total_count = col_count * row_count

            # get the original image metadata and update it with the new height and width of the tiles.
            profile = src.profile
            profile.update(width = const.TILE_SIZE, height = const.TILE_SIZE, count = imarray.shape[0]-1)

            # Extract the original coordinates of the image as well as the pixel column & row strides.
            original_affine = src.affine
            pixel_col_stride = original_affine[0]
            pixel_row_stride = original_affine[4]

            # Loop through all columns and rows.
            for row_no in xrange(row_count):
                for col_no in xrange(col_count):
                    current_tile = (row_no*col_count + col_no)
                    if current_tile%50 == 0:
                        print "Created %d of %d tiles"%(current_tile, total_count)

                    # calculate the new coordinates of the tile
                    transformation_matrix = \
                        rasterio.Affine.translation(xoff=col_no*pixel_col_stride * const.TILE_SIZE,
                                                    yoff=row_no*pixel_row_stride* const.TILE_SIZE)
                    tile_affine = transformation_matrix * original_affine

                    # update the tile metadata with the new coordinates
                    profile.update(affine = tile_affine, transform = tile_affine)

                    # define the pixel coordinates of the tile and get the respective image chunk
                    read_window = (row_no * const.TILE_SIZE, (row_no+1) * const.TILE_SIZE), \
                                  (col_no * const.TILE_SIZE, (col_no+1) * const.TILE_SIZE)
                    current_tile = src.read(window = read_window)

                    # save the tile and origin image in the tile directory.
                    # Check in the last band of the tile (which is the pixel mask), if the tile contains any whitespace
                    # If it contains any whitespace, throw it out!
                    if current_tile[-1,0,0] and \
                            current_tile[-1, const.TILE_SIZE - 1,0] and \
                            current_tile[-1,0, const.TILE_SIZE-1] and \
                            current_tile[-1, const.TILE_SIZE-1, const.TILE_SIZE-1]:
                        with rasterio.open(tiles_dir_path + '/tile_' + str(row_no) + ',' + str(col_no) + '.tif', 'w', **profile) as dst:
                            # save all but the mask dimension to the tile
                            dst.write(current_tile[0:-1, :, :])
                        with open(tiles_dir_path+'/tile_' + str(row_no) + ',' + str(col_no) + '.txt', "w") as file:
                            file.write(os.path.basename(image_file_path))

            print "Finished tiling %d tiles"%total_count

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-i', '--imagePath', dest='image_path', default=const.ORIGINALS_DIR)
    parser.add_option('-t', '--tilesPath', dest = 'tiles_path', default = const.TILES_DIR)

    (options, _) = parser.parse_args()

    tiles_from_image(options.image_path, options.tiles_path)


