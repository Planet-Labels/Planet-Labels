__author__ = 'vincent'

import numpy as np
import h5py
import os
import rasterio
from skimage.filters import gabor_kernel
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
import matplotlib.colors
import const

class FeatureExtractor():
    '''
    Abstract class of feature extractors.
    '''
    # The file ending used to indicate this feature
    ending = ''
    # The length of the feature vector when this feature extractor is used
    feature_length = 0

    def center_tile(self, imarray):
        '''
        Removes the mean from an imarray.

        :param imarray: A numpy array of shape (c, h, w) with number of channels c, height h and width w
        :returns: The centered numpy array.
        '''
        mean = np.mean(imarray, axis = 0) # Compute mean of the entire image
        imarray_centered = imarray - mean

        return imarray_centered

    def extract_image_features(self, imarray):
        raise NotImplementedError("extract_image_feature is an abstract method.")


class FE_RGB_Pixel_Values(FeatureExtractor):
    '''
    Extracts the R,G,B raw pixel values into a 1D-numpy array.
    '''
    ending = '.pix'
    feature_length = (const.TILE_SIZE**2)*3

    def extract_image_features(self, imarray):
        imarray = self.center_tile(imarray[0:3, :, :])
        (c, h, l) = imarray.shape
        result = np.zeros((c, h*l)) # Storing all the layers as a single 1D vector

        for i in xrange(c):
            result[i] = imarray[i].flatten()

        return np.ndarray.flatten(result)


class FE_Gabor_Filters(FeatureExtractor):
    '''
    Uses a gabor filter bank to extract texture information from the image.
    Adaptation of http://scikit-image.org/docs/dev/auto_examples/plot_gabor.html
    '''
    ending = '.gab'

    angles = range(4)
    frequencies = [0.05, 0.25]
    scales = [1, 3]

    feature_length = len(angles) * len(scales) * len(frequencies) * 2

    debugging_kernels = []
    debugging_results = []

    kernels = []

    def __init__(self):
        for angle in self.angles:
            angle = angle/4.0 * np.pi
            for frequency in self.frequencies:
                self.debugging_kernels.append(gabor_kernel(frequency=frequency, theta=angle))

                for scale in self.scales:
                    kernel = np.real(gabor_kernel(frequency = frequency, theta = angle, sigma_x = scale, sigma_y = scale))
                    self.kernels.append(kernel)

    def power(self, image, kernel):
    # Normalize images for better comparison.
        image = (image - image.mean()) / image.std()
        result= np.sqrt(ndi.convolve(image, np.real(kernel), mode='wrap')**2 +
                       ndi.convolve(image, np.imag(kernel), mode='wrap')**2)
        return result

    def visualize(self, black_white_image):
        fig, axes = plt.subplots(nrows=len(self.debugging_results)+1, ncols=2, figsize=(5, 6))
        plt.gray()

        fig.suptitle('Image responses for Gabor filter kernels', fontsize=12)

        axes[0][0].axis('off')

        # Plot original images
        axes[0][1].imshow(black_white_image)
        axes[0][1].set_title("Satellite Image", fontsize=9)
        axes[0][1].axis('off')

        for (kernel, power), ax_row in zip(self.debugging_results, axes[1:]):
            # Plot Gabor kernels
            ax = ax_row[0]
            ax.imshow(np.real(kernel), interpolation='nearest')
            ax.set_xticks([])
            ax.set_yticks([])

            # Plot Gabor responses with the contrast normalized for each filter
            vmin = np.min(power)
            vmax = np.max(power)

            ax_row[1].imshow(power, vmin=vmin, vmax=vmax)
            ax.axis('off')

        plt.show()

    def extract_image_features(self, imarray):
        # Transpose the imarray into standard [1,1,3] shape
        imarray_trans_cent = np.transpose(imarray, (1, 2, 0))

        # convert the image into a blackwhite image
        imarray_bw = np.mean(imarray_trans_cent, axis = 2)

        # Center the BW image
        imarray_bw_cent = imarray_bw - np.mean(imarray_bw)

        feature_vector = np.zeros((len(self.kernels), 2), dtype=np.double)
        for i, kernel in enumerate(self.kernels):
            # Convolve the centered bw image with current kernel and extract mean and variance of the intensities as features.
            filtered = ndi.convolve(imarray_bw_cent, kernel, mode='wrap')
            feature_vector[i, 0] = np.mean(filtered)
            feature_vector[i, 1] = np.var(filtered)

        # flatten the feature vector to 1D.
        feature_vector = feature_vector.flatten()

        if const.DEBUG_FES:
            for i, kernel in enumerate(self.debugging_kernels):
                self.debugging_results.append((kernel, self.power(imarray_bw, kernel)))

            self.visualize(imarray_bw)

        return feature_vector

class FE_RGB_Histograms(FeatureExtractor):
    '''
    Extracts R,G,B histograms of each picture with 50 bins each.
    '''
    ending = '.rgb_hist'
    feature_length = 150

    def extract_image_features(self, imarray):
        imarray_rgb = imarray[0:3, :, :]
        imarray_rgb_centered = self.center_tile(imarray_rgb)

        (c, h, l) = imarray_rgb_centered.shape
        color_histograms = np.zeros((c, 50))

        for i in xrange(c):
            edges, color_histograms[i, :] = np.histogram(imarray_rgb_centered[i, :, :], bins = 49)

        return np.ndarray.flatten(color_histograms)

class FE_HSV_Histograms(FeatureExtractor):
    '''
    Extracts H,S,V histograms of each picture with 50 bins for H,S,V respectively.
    '''
    ending = '.hsv_hist'
    feature_length = 150

    def extract_image_features(self, imarray):
        imarray_rgb = imarray[0:3, :, :]
        imarray_transposed = np.transpose(imarray_rgb, (1, 2, 0 ))
        imarray_hsv = matplotlib.colors.rgb_to_hsv(imarray_transposed)

        (l, h, c) = imarray_hsv.shape
        color_histograms = np.zeros((c, 50))

        for i in xrange(c):
            edges, color_histograms[i, :] = np.histogram(imarray_hsv[:, :, i], bins = 49)

        if const.DEBUG_FES:
            plt.imshow(imarray_hsv[:,:,0], cmap='hsv')
            plt.show()

        return np.ndarray.flatten(color_histograms)

class FE_RedEdge_NIR_Histograms(FeatureExtractor):
    '''
    Extracts Red Edge and Near-Infrared histograms
    '''
    feature_length = 100
    ending = '.redg_nir'

    def extract_image_features(self, imarray):
        imarray_redg_nir = imarray[-2:, :, :]
        imarray_redg_nir_centered = self.center_tile(imarray_redg_nir)

        (c, h, l) = imarray_redg_nir_centered.shape
        histograms = np.zeros((c, 50))

        for i in xrange(c):
            edges, histograms[i, :] = np.histogram(imarray_redg_nir_centered[i, :, :], bins = 49)

        if const.DEBUG_FES:
            imarray_transposed = np.transpose(imarray_redg_nir, (1, 2, 0 ))
            plt.imshow(imarray_transposed[:,:,0], cmap='hsv')
            plt.show()

        return np.ndarray.flatten(histograms)

class JointFeatureExtractor():
    '''
    Takes a list of classes derived from the FeatureExtractor class and concatenates their feature vectors.
    '''

    FE_ids = {
        'gab': FE_Gabor_Filters,
        'rgb_hist': FE_RGB_Histograms,
        'ir_re_hist': FE_RedEdge_NIR_Histograms,
        'hsv_hist': FE_HSV_Histograms,
        'rgb_pix': FE_RGB_Pixel_Values
    }

    def __init__(self, tile_dir, requested_FE_ids):
        self.feature_extractors = [self.FE_ids[id]() for id in requested_FE_ids]
        self.feature_length = sum([feature_extractor.feature_length for feature_extractor in self.feature_extractors])
        self.ending = "." + "_".join([feature_extractor.ending[1:] for feature_extractor in self.feature_extractors])
        self.tile_dir = tile_dir

        print "Initialized the feature extractor. Using the following features: "
        print requested_FE_ids
        print "Feature vector length is %d"%self.feature_length


    def generate_feature_files(self):
        '''
        Crawls the tile path, extracts features from tiles and writes them into the respective tile directory.
        :return:
        '''
        for tile_no in [name for name in os.listdir(self.tile_dir) if os.path.isdir(os.path.join(self.tile_dir, name))]:
            if not os.path.exists(os.path.join(self.tile_dir, tile_no, tile_no + self.ending)):
                file_name = tile_no + str(".tif")
                full_file_path = os.path.join(self.tile_dir, tile_no, file_name)
                print "Extracting features of tile ", tile_no

                # Load the tile into memory
                imarray = self.open_tile(full_file_path)

                # Extract the feature vectors
                feature_vector_list = []

                # The image is passed to each feature extractor. The resulting list is concatenated into one
                # large feature vector.
                for feature_extractor in self.feature_extractors:
                    feature_vector_list.append(feature_extractor.extract_image_features(imarray))

                feature_vector_np_array = np.hstack(feature_vector_list)

                if (len(feature_vector_np_array) == self.feature_length):
                    # Write the feature vector into a file in the tile directory
                    self.write_feature_file(tile_no, feature_vector_np_array)
                else:
                    print "The length of the feature vector is incorrect. The actual lenght is %d"%len(feature_vector_np_array)

    def write_feature_file(self, tile_no, feature_vector):
        with h5py.File(os.path.join(self.tile_dir, tile_no, tile_no + self.ending), "w") as file:
            file['features'] = feature_vector

    def open_tile(self, full_file_path):
        with rasterio.open(full_file_path) as src:
            return src.read()

    def generate_single_feature_file(self, image_path):
        print "Extracting features of file ", image_path

        # Load the tile into memory
        imarray = self.open_tile(image_path)

        # Extract the feature vectors
        # The image is passed to each feature extractor. The resulting list is concatenated into one
        # large feature vector.
        feature_vector_list = []
        for feature_extractor in self.feature_extractors:
            feature_vector_list.append(feature_extractor.extract_image_features(imarray))

        feature_vector_np_array = np.hstack(feature_vector_list)

        assert (len(feature_vector_np_array) == self.feature_length)

        # Write the feature vector into a file in the tile directory
        file_name = os.path.join(self.tile_dir, os.path.splitext(image_path)[0] + self.ending)
        with h5py.File(file_name, "w") as file:
            file['features'] = feature_vector_np_array

        return file_name

