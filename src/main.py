__author__ = 'timon'

import optparse
import os
import datetime

import utils.data_setup
import learning.training_preparation as training_preparation
import learning.error_analysis as error_analysis
import learning.data_analysis as da
import learning.Classifiers
import utils.const

############################################################################
#
#   PlanetLabels by Rishabh Bhargava, Vincent Sitzmann and Timon Ruban
#
#   Main Data Pipeline:
#
#           1. Data Setup
#                   - Download Images
#                   - Split into tiles
#                   - Extract utils.const.FEATURES
#           2. Training
#           3. Prediction
#           4. Error Analysis
#
############################################################################

def main(options):
    # data setup
    (original_dir, train_dir, validation_dir, test_dir, now_dir, nlcd) = check_directories(options.path)

    # Downloading a set of example images from the planet labs explorer program. API key in utils.const necessary!
    if options.download == True:
        print "Downloading Pictures..."
        utils.data_setup.download_pictures(original_dir)
    # The images in the "originals" directory are tiled.
    if options.tile == True:
        print "Tiling Pictures..."
        utils.data_setup.make_tiles(original_dir, train_dir, nlcd)
    # From each tile, a feature vector is extracted.
    if options.extract == True:
        print "Extracting Features..."
        utils.data_setup.extract_features(train_dir)
    # The dataset is automatically split into a training, validation and testing set.
    if options.split == True:
        print "Splitting Dataset in Train, Validation and Testset..."
        utils.data_setup.split_data(train_dir, validation_dir, test_dir)

    # A histogram of the labels in the training dataset is computed.
    print "\nComputing label histogram of data..."
    file_name = "label_histogram.hdf5"
    file_path = os.path.join(now_dir, file_name)
    da.write_label_histogram(train_dir, file_path)

    print "\nReading in the training data..."
    (X_train, y_train) = training_preparation.read_data(train_dir, utils.const.FEATURES, utils.const.NUM_FEATURES)

    # The model used is a random forest classifier.
    model = learning.Classifiers.RandomForest_c()
    print "Using: " + model.description()

    print "\nTraining the model..."
    model.train(X_train, y_train)

    # The model is serialized with python's pickle and saved for later usage.
    print "\nSaving the model..."
    model_name = utils.const.METHOD + "_" + utils.const.FEATURES + "_model.p"
    model_path = os.path.join(now_dir, model_name)
    model.write(model_path)

    # The model is validated on the validation set.
    print "Reading in the validation data..."
    (X_test, y_test) = training_preparation.read_data(validation_dir, utils.const.FEATURES, utils.const.NUM_FEATURES)
    print "\nPredicting the labels..."
    y_pred = model.predict(X_test)

    # Error statistics are computed.
    print "Computing the confusion matrix..."
    file_name_cm = utils.const.METHOD + "_" + utils.const.FEATURES + "_confusion_matrix.hdf5"
    file_path_cm = os.path.join(now_dir, file_name_cm)
    error_analysis.write_confusion_matrix(y_pred, y_test, file_path_cm)

    print "Computing precision and recall..."
    file_name_pr = utils.const.METHOD + "_" + utils.const.FEATURES + "_precision_recall.json"
    file_path_pr = os.path.join(now_dir, file_name_pr)
    test_error = error_analysis.write_prec_recall(file_path_cm, file_path_pr)
    print "Plotting the confusion matrix..."
    error_analysis.plot_confusion_matrix(file_path_cm)

    print "Computing the learning curve..."
    file_name = utils.const.METHOD + "_" + utils.const.FEATURES + "_learning_curve.hdf5"
    file_path = os.path.join(now_dir, file_name)
    error_analysis.write_learning_curves(X_train, X_test, y_train, y_test,  model, file_path)
    print "Plotting the learning curve..."
    error_analysis.plot_learning_curves(file_path)

    print "Training of the model is finished. It achieved a test error of %f. The path to the full model is: \n%s"%(test_error, model_path)

def check_directories(main_folder):
    '''
    sets up and checks the necessary directory structure

    :param main_folder: The root folder, so far only containing a folder "NLCD" and a folder "originals".
    :return: A tuple of the paths to the created directories.
    '''
    original_dir = os.path.join(main_folder, utils.const.ORIGINALS_DIR)
    train_dir = os.path.join(main_folder, utils.const.TRAIN_DIR)
    validation_dir = os.path.join(main_folder, utils.const.VALIDATION_DIR)
    test_dir = os.path.join(main_folder, utils.const.TEST_DIR)
    analyis_dir = os.path.join(main_folder, utils.const.ANALYSIS_DIR)
    now_dir = os.path.join(analyis_dir, ''.join(datetime.datetime.now().isoformat().split('.')[0].split(':')))
    nlcd_dir = os.path.join(main_folder, utils.const.NLCD_DIR)

    # check if directories exist. If not, create them.
    for dir in [original_dir, train_dir, validation_dir, analyis_dir, nlcd_dir, now_dir]:
        if not os.path.isdir(dir):
            print "Creating directory %s" % (dir)
            os.mkdir(dir)

    # check nlcd file is present. If not, issue a warning!
    nlcd = os.path.join(nlcd_dir, utils.const.NLCD2011)
    if not os.path.isfile(nlcd):
        print "!!!!!!!!!!!!!!!!!!"
        print "No NLCD File found"
        print "!!!!!!!!!!!!!!!!!!"

    return (original_dir, train_dir, validation_dir, test_dir, now_dir, nlcd)

if __name__ == "__main__":
    parser = optparse.OptionParser()
    parser.add_option('-p', '--path', dest='path') # abs. path to folder that will hold data folder
    parser.add_option('-d', '--download', dest='download', default=False, action='store_true')
    parser.add_option('-t', '--tile', dest='tile', default=False, action='store_true')
    parser.add_option('-e', '--extract', dest='extract', default=False, action='store_true')
    parser.add_option('-s', '--split', dest='split', default=False, action='store_true')

    (options, _) = parser.parse_args()
    main(options)

