import h5py
import matplotlib.pyplot as plt
import numpy as np
import json
from sklearn.metrics import confusion_matrix

def get_error(X, y, model):
    """
    Predict labels for X and report error using y

    :param X: np.array data matrix with features
    :param y: np.array corresponding classifications
    :param model: trained model
    :return: error as a percentage
    """
    y_pred = model.predict(X)

    # Count number of errors
    count = 0
    for i, a in enumerate(y_pred):
        if a != y[i]:
            count += 1
    error = 1.0*count/len(y_pred)

    return error

def write_confusion_matrix(y_pred, y_test, file_path):
    """Compute the confusion matrix and write top output"""
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    # write to file
    label_dict = {10: "Water", 20:"Developed",
                  30:"Barren", 40:"Forest", 50:"Shrubland",
                  70:"Herbaceous", 80:"Cultivated", 90:"Wetlands"}
    labels = [str(label_dict[number]) for number in sorted(label_dict.keys()) if (number in y_test or number in y_pred)]
    print "Writing confusion matrix to file..."
    with h5py.File(file_path, "w") as file:
        file['labels'] = labels
        file['confusion_matrix'] = cm
        file['confusion_matrix_n'] = cm_normalized

def write_prec_recall(file_path_cm, file_path_pr):
    """Compute the precision and recall"""
    with h5py.File(file_path_cm, 'r') as data_file:
        labels = data_file['labels'][()]
        cm = data_file['confusion_matrix'][()]

    prec_recall_dict = {}
    for i, label in enumerate(labels):
        precision = cm[i,i] / float(np.sum(cm[:,i]))
        recall = cm[i,i] / float(np.sum(cm[i,:]))
        prec_recall_dict[label] = (precision, recall)

    # test error
    correct_pred = np.sum(cm.diagonal()) / float(np.sum(cm))
    test_error = 1 - correct_pred
    prec_recall_dict['test_error'] = test_error

    with open(file_path_pr, 'w') as outfile:
        json.dump(prec_recall_dict, outfile, sort_keys=True, indent=4)

    return test_error

def plot_confusion_matrix(file_path):
    """Plot the confusion matrix using data found at file_path"""
    with h5py.File(file_path, 'r') as data_file:
        labels = data_file['labels'][()]
        cm = data_file['confusion_matrix_n'][()]
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=90)
    plt.yticks(tick_marks, labels)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def write_learning_curves(X_train, X_test, y_train, y_test, model, file_path):
    """Compute the learning curve and write to output"""
    test_errors = []
    train_errors = []
    max_set_size = X_train.shape[0]
    set_sizes = [(i*max_set_size)/5 for i in range(1,6)]
    for set_size in set_sizes:
        print 'Training set size: ', set_size
        model.train(X_train[:set_size, :], y_train[:set_size])
        test_error = get_error(X_test, y_test, model)
        train_error = get_error(X_train[:set_size, :], y_train[:set_size], model)
        print "Validation set error: " + str(test_error)
        print "Training error: " + str(train_error)
        test_errors.append(test_error)
        train_errors.append(train_error)
    print ""

    # write to file
    print "Writing training and test errors to file..."
    with h5py.File(file_path, "w") as file:
        file['train_error'] = train_errors
        file['test_error'] = test_errors
        file['size'] = set_sizes

def plot_learning_curves(file_path):
    """Plot the learning curve using the data found at file_path"""
    with h5py.File(file_path, 'r') as data_file:
        train_errors = data_file['train_error'][()]
        test_errors = data_file['test_error'][()]
        set_sizes = data_file['size'][()]
    # plot
    plt.style.use('ggplot')
    plt.figure()
    plt.title('Learning')
    plt.xlabel('Training set size')
    plt.ylabel('Error')
    plt.ylim([0,1])
    plt.plot(set_sizes, test_errors, 'rs', label="Test error", linestyle = '-')
    plt.plot(set_sizes, train_errors, 'b^', label="Training error", linestyle = '-')
    leg = plt.legend(loc="upper right", shadow=True)
    leg.get_frame().set_facecolor("#FFFFFF")
    plt.show()

