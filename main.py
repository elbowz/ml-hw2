#!/usr/bin/env python3

'''
# How to install dependencies on Arch Linux
sudo pip install -U scikit-learn
sudo pip install -U matplotlib
sudo pip install -U pandas
sudo pacman -S tk
'''

import warnings

warnings.filterwarnings("ignore")

import sys
import itertools
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, KFold
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from pprint import pprint

plt.rcParams.update({'font.size': 8})

__default_ts_name = 'wine'
__default_test_size = 0.20

__default_ts_normalizing = True
__default_ts_one_hot_encoding = True

__default_training_verbosity = False

__default_training_k_fold = False
__default_training_k_fold_iteration = 10
__default_hidden_layer_k_fold = 1600
__default_input_encoding_k_fold = 1     # 0, 1, 2

__default_train_test_split_random_state = 90

__default_save_img = False
__default_graph_for_each_hidden_layer = True
__default_hidden_layer_1 = (2, 10, 60, 160, 300, 800, 1600, 3000, 10000)
__default_hidden_layer_2 = ((4, 2), (8, 4), (18, 8), (40, 18), (100, 40), (300, 100), (1000, 400))
__default_hidden_layer_3 = ((4, 3, 2), (10, 6, 4), (30, 10, 4), (100, 80, 30),
                            (300, 200, 100), (800, 600, 400), (1000, 800, 400))
__default_hidden_layer_4 = ((6, 4, 3, 2), (10, 6, 4, 2), (30, 10, 4, 2),
                            (100, 80, 30, 10), (300, 200, 100, 50), (800, 600, 400, 200))

__default_hidden_layer_full = __default_hidden_layer_1 + __default_hidden_layer_2 + __default_hidden_layer_3 + __default_hidden_layer_4

__ts_opts = {
    'wine': {
        'url': "./ts/wine.data",
        'columns': (
            'Class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids',
            'Nonflavanoid phenols', 'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
            'Proline'),
        'classes_name': ['1', '2', '3'],
        'x_slice': (slice(None, None), slice(1, None)),
        'y_slice': (slice(None, None), 0),
        'hidden_layers_sizes': __default_hidden_layer_full,
    },
    'breast-cancer': {
        'url': "./ts/breast-cancer-wisconsin.data",
        'columns': ('Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                    'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin',
                    'Normal Nucleoli',
                    'Mitoses', 'Class'),
        'classes_name': ['benign', 'malignant'],
        'x_slice': (slice(None, None), slice(1, -1)),
        'y_slice': (slice(None, None), -1),
        'hidden_layers_sizes': __default_hidden_layer_full,
    },
    'letters': {
        'url': "./ts/letter-recognition.data",
        'columns': (
            'lettr', 'x-box', 'y-box', 'width', 'high', 'onpix', 'x-bar', 'y-bar', 'x2bar', 'y2bar', 'xybar', 'x2ybr',
            'xy2br', 'x-ege', 'xegvy', 'y-ege', 'yegvx'),
        'classes_name': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O',
                         'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z'],
        'x_slice': (slice(None, None), slice(1, None)),
        'y_slice': (slice(None, None), 0),
        'hidden_layers_sizes': __default_hidden_layer_full,

    },
    'poker': {
        'url': "./ts/poker-hand-testing.data",
        'columns': ('S1', 'C1', 'S2', 'C2', 'S3', 'C3', 'S4', 'C4', 'S5', 'C5', 'Class'),
        'x_slice': (slice(None, None), slice(0, -1)),
        'y_slice': (slice(None, None), -1),
        'classes_name': ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
        'hidden_layers_sizes': __default_hidden_layer_full,
    },
}


def main(argv):
    if len(argv) > 1 and argv[1] in ['-h', '--help']:
        print('Usage: {} [wine|breast-cancer|letters|poker][test_size < 1 | k-fold > 1][-v]\n'.format(argv[0]))
        exit(0)

    trainingset_selected_name = __default_ts_name
    test_size = __default_test_size
    training_k_fold = __default_training_k_fold
    training_k_fold_iteration = __default_training_k_fold_iteration
    training_verbosity = __default_training_verbosity

    if len(argv) > 1 and argv[1] in __ts_opts:
        trainingset_selected_name = argv[1]

        if len(argv) > 2:
            if float(argv[2]) < 1:
                training_k_fold = False
                test_size = float(argv[2])
            else:
                training_k_fold = True
                training_k_fold_iteration = int(argv[2])

            training_verbosity = (len(argv) == 4 and argv[3] == '-v')

    ts_selected_opts = __ts_opts[trainingset_selected_name]

    print('\nTrainingSet selected: ' + ts_selected_opts['url'])

    # Read dataset to pandas dataframe
    dataset = pd.read_csv(ts_selected_opts['url'], names=ts_selected_opts['columns'])

    print('\nFirst five rows of TrainingSet:\n')
    print(dataset.head())

    # Remove row with question marks
    dataset = dataset[~(dataset.astype(str) == '?').any(1)]

    print('\nDataSet Length: {}'.format(len(dataset)))

    # Select Input/features (columns) and Output (columns), slicing the dataset (matrix)
    X = dataset.iloc[ts_selected_opts['x_slice'][0], ts_selected_opts['x_slice'][1]]
    y = dataset.iloc[ts_selected_opts['y_slice'][0], ts_selected_opts['y_slice'][1]]

    # print('\nInput:\n')
    # print(X.head())
    # print('\nOutput:\n')
    # print(y.head())

    X = X.values
    y = y.values

    X_encoded_lst = [X]
    X_encoded_labels = ['no-encoding']

    # Normalize the Features (center and scaling them)
    if __default_ts_normalizing:
        print('Encoded input: Center and scaling')
        X_encoded_labels.append('scaler')
        X_encoded_lst.append(ts_encoding(X, 'scaling'))

    # Trasform Features in characteristic vector (cannot combine with Normalization)
    if __default_ts_one_hot_encoding:
        print('Encoded input: Characteristic vectors')
        X_encoded_labels.append('characteristic vector')
        X_encoded_lst.append(ts_encoding(X, 'onehot'))

    if not training_k_fold:

        print('### Running normal test on {} (test size: {})'.format(trainingset_selected_name, test_size))

        accuracy_by_encoded = []

        for index, X_encoded in enumerate(X_encoded_lst):

            print('\n## {}: {})'.format(X_encoded_labels[index], X_encoded[0]))

            # Normal training (no K-folding)
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=test_size,
                                                                random_state=__default_train_test_split_random_state)

            accuracy_by_hidden_layer_sizes = []

            for hidden_layer_sizes in ts_selected_opts['hidden_layers_sizes']:
                print('# {:<20} => '.format(str(hidden_layer_sizes)), end='')

                # Init MLP
                classifier = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, verbose=training_verbosity,
                                           max_iter=1000)

                classifier.fit(X_train, y_train)
                y_pred = classifier.predict(X_test)

                accuracy = print_results(classifier, y_test, y_pred, ts_selected_opts['classes_name'], plot_cm=False,
                                         print_cm=False, print_report=False)
                accuracy_by_hidden_layer_sizes.append(accuracy)

            accuracy_by_encoded.append(accuracy_by_hidden_layer_sizes)

        # Hidden layer size to string for X axis
        hidden_layers_sizes_str = [str(hidden_layer) for hidden_layer in ts_selected_opts['hidden_layers_sizes']]

        print('\nHidden layers size: {}'.format(hidden_layers_sizes_str))
        print('\nAccuracy: {}'.format(accuracy_by_encoded))

        # Plot a graph for each hidden layers deep category
        if __default_graph_for_each_hidden_layer:
            ldhs = [0]
            ldhs.append(len(__default_hidden_layer_1))
            ldhs.append(ldhs[1] + len(__default_hidden_layer_2))
            ldhs.append(ldhs[2] + len(__default_hidden_layer_3))
            ldhs.append(ldhs[3] + len(__default_hidden_layer_4))

            ldhs_slices = (slice(ldhs[0], ldhs[1]), slice(ldhs[1], ldhs[2]), slice(ldhs[2], ldhs[3]), slice(ldhs[3], ldhs[4]))

            for index, lhds_slice in enumerate(ldhs_slices):
                layer = [accuracy_by_encoded_l1[lhds_slice] for accuracy_by_encoded_l1 in accuracy_by_encoded]

                plot_line_graph(hidden_layers_sizes_str[lhds_slice], layer, labels=X_encoded_labels,
                                title='Accuracy on {} hidden layers size and different encoding ({})'
                                .format(index + 1, trainingset_selected_name),
                                xlabel='Hidden layer size',
                                ylabel='Precision')

                if __default_save_img:
                    plt.savefig('{}-layer{}-{}.png'.format(trainingset_selected_name, index+1, id_generator()), dpi=200)

        # Plot a graph whole hidden layers
        plot_line_graph(hidden_layers_sizes_str, accuracy_by_encoded, labels=X_encoded_labels,
                        title='Accuracy on different hidden layer size and encoding ({})'
                        .format(trainingset_selected_name),
                        xlabel='Hidden layer size',
                        ylabel='Precision')

    else:
        plt.rcParams.update({'font.size': 14})

        X_encoded = X_encoded_lst[__default_input_encoding_k_fold]

        print('### Running {}-fold on {}\n'.format(training_k_fold_iteration, trainingset_selected_name))
        print('# {} | {}'.format(str(__default_hidden_layer_k_fold), X_encoded_labels[__default_input_encoding_k_fold]))

        classifier = MLPClassifier(hidden_layer_sizes=__default_hidden_layer_k_fold, verbose=training_verbosity, max_iter=500)

        # K-folding
        accuracy_k_fold = []

        k_fold = KFold(n_splits=training_k_fold_iteration, shuffle=True)

        for train_index, test_index in k_fold.split(X_encoded):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            classifier.fit(X_train, y_train)
            y_pred = classifier.predict(X_test)

            accuracy = print_results(classifier, y_test, y_pred, ts_selected_opts['classes_name'])
            accuracy_k_fold.append(accuracy)

        accuracy_avg = sum(accuracy_k_fold) / float(len(accuracy_k_fold))
        print('Precision AVG: {:.4f}\n'.format(accuracy_avg))

        plot_line_graph(range(training_k_fold_iteration), [accuracy_k_fold], labels=('precision',),
                        title='Accuracy by {}-fold ({})'.format(training_k_fold_iteration, trainingset_selected_name),
                        xlabel='k-fold',
                        ylabel='Precision')

    if __default_save_img:
        plt.savefig('{}-{}.png'.format(trainingset_selected_name, id_generator()), dpi=200)

    if not __default_save_img:
        plt.show()

    from sklearn.externals import joblib
    # save the model to disk
    # filename = 'finalized_model.sav'
    # joblib.dump(model, filename)
    #
    # # some time later...
    #
    # # load the model from disk
    # loaded_model = joblib.load(filename)
    # result = loaded_model.score(X_test, Y_test)
    # print(result)


def ts_encoding(x, type='scaling'):
    # Normalize the Features (center and scaling them)
    if type == 'scaling':
        data_scaler = StandardScaler()

        # Fit train data
        data_scaler.fit(x)
        return data_scaler.transform(x)

    # Trasform Features in carateristic vectors (cannot combine with Normalization)
    elif type == 'onehot':
        encoder = OneHotEncoder(handle_unknown='ignore')
        encoder.fit(x)

        # print(encoder.categories_)
        return encoder.transform(x).toarray()


def print_results(classifier, y_test, y_pred, classes, plot_cm=True, print_cm=True, print_report=True):
    """
    Print full report after a test
    """
    accuracy = accuracy_score(y_test, y_pred)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=classes, labels=list(classifier.classes_))

    print('Precision: {:<6.4f} (epochs: {:>4} | out: {:>2} | func: {:>8})'.format(accuracy, classifier.n_iter_, classifier.n_outputs_, classifier.out_activation_))

    if print_cm:
        print('Confusion matrix:')
        matrix_print(cm)

    if print_report:
        print()
        print('Report: \n{}'.format(report))

    if plot_cm:
        plot_confusion_matrix(cm, title='Confusion matrix', classes=classes)

    return accuracy


def plot_line_graph(x, ylist, fmt=('bo-', 'g^-', 'rv-', 'c>:', 'm<-'), labels=None, title='', xlabel='', ylabel='', ymax=1,
                    figsize=(12, 6)):
    """
    Plot multiple line on single graph
    """
    plt.figure(figsize=figsize)
    plt.ylim(top=ymax, bottom=min([item for sublist in ylist for item in sublist]))

    for index, y in enumerate(ylist):
        label = labels[index] if labels is not None else None
        plt.plot(x, y, fmt[index], label=label, linewidth=3, markersize=12)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)

    plt.legend(loc="best")


def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    Plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(8, 8))

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


import string
import random

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def matrix_print(mat, fmt="g"):
    """
    Pretty matrix print
    """
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


import datetime

def get_time(seconds):
    return str(datetime.timedelta(seconds=seconds))


import time

if __name__ == "__main__":
    start_time = time.time()
    main(sys.argv)
    print("--- %s (running time) ---" % (get_time(time.time() - start_time)))

    exit(0)
