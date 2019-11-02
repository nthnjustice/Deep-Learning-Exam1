# load dependencies
import os
import numpy as np
from PIL import Image
from PIL import ImageFilter
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint


# description: implementation of multi-layer perceptron neural network
class MLP:
    def __init__(self):
        # initialize storage for data
        self.images = []
        self.labels = []

        # initialize storage for train-test data splits
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        # initialize storage for model instance and model score
        self.model = None
        self.score = None

    # purpose: load, preprocess, and store training data
    # input: 1) tuple for desired size images are scaled to (must be integer value if padding is applied)
    #        2) boolean value for whether or not white-cell padding is applied to images
    #        3) boolean value for whether or not a PIL filter is applied to images
    #        4) boolean value for whether or not images should be converted to grayscale
    # effect: populates properties for images as arrays and corresponding class labels
    def load_data(self, resize, apply_padding, apply_filter, apply_grayscale):
        # initialize storage with path names for images
        image_files = [file for file in os.listdir("data/train") if file.endswith(".png")]

        # loop through image paths
        for i in image_files:
            # capture image in focus
            image = Image.open("data/train/" + i)

            if apply_padding:
                # https://jdhao.github.io/2017/11/06/resize-image-to-square-with-padding/

                # calculate proportion of white-cell padding for image in focus
                ratio = float(resize) / max(image.size)
                size = tuple([int(x * ratio) for x in image.size])
                temp_image = image.resize(size, resample=Image.NEAREST)

                # re-create image in focus with new size and padding
                image = Image.new("RGB", (resize, resize), (255, 255, 255))
                image.paste(temp_image, ((resize - size[0]) // 2, (resize - size[1]) // 2))
            else:
                # resize image in focus
                image = image.resize(resize, resample=Image.NEAREST)

            if apply_filter:
                # apply filter to image in focus
                image = image.filter(ImageFilter.BLUR)

            if apply_grayscale:
                # https://www.science-emergence.com/Articles/How-to-convert-an-image-to-grayscale-using-python-/

                # convert image in focus to grayscale
                image = image.convert("LA")

            # convert image in focus to array
            image = np.asarray(image)
            # store the processed image in focus
            self.images.append(image)

        # initialize storage with path names for class labels
        text_files = [file for file in os.listdir("data/train") if file.endswith(".txt")]

        # loop through class label paths
        for i in text_files:
            # capture class label in focus
            label = open("data/train/" + i, "r")
            label = label.read()
            # store the class label in focus
            self.labels.append(label)

    # purpose: apply preprocessing methods to build model train-test split
    # input: 1) boolean value for whether or not to apply minmax scaling (False => division by 255)
    #        2) string with value "over" for oversampling or "under" for undersampling method application
    # effect: builds train-test split with scaling, sampling, and encoding methods applied
    def prep_data(self, minmax, sampler):
        # convert image and class label storage to arrays
        x = np.array(self.images)
        y = np.array(self.labels)

        # https://github.com/amir-jafari/Deep-Learning/tree/master/Exam_MiniProjects/3-Keras_Exam1_Sample_Codes_F19

        # flatten image array
        x = x.reshape(len(x), -1)

        if minmax:
            # apply minmax scaling to images array
            scaler = MinMaxScaler()
            x = scaler.fit_transform(x)
        else:
            # divide image arrays by 255 (converts RBG values into 0-1 range)
            x = x / 255

        if sampler == "over":
            # apply oversampling method to input and target data
            ros = RandomOverSampler(random_state=0)
            x, y = ros.fit_sample(x, y)
        elif sampler == "under":
            # apply undersampling method to input and target data
            rus = RandomUnderSampler(random_state=0)
            x, y = rus.fit_resample(x, y)

        # encode class label target values to integers and categorize them for Keras usage
        le = LabelEncoder()
        y = le.fit_transform(y)
        y = to_categorical(y, num_classes=4)

        # build train-test data split
        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.30, stratify=y)

        # store train-test data split
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    # purpose: build and run Keras implementation of MLP
    # inputs: 1) integer value for the number of neurons
    #         2) string value for the activation function to use
    #         3) float value for the dropout rate to use
    #         4) string value for the optimization method to use
    #         5) string value for the loss function to use
    #         6) string value for the metric type to score on
    #         7) integer value for the batch size to use
    #         8) integer value for the number of epochs to use
    # effect: builds and saves MLP model
    def train_model(self, neurons, activation, dropout, optimizer, loss, metrics, batch_size, epochs):
        # copy train-test split for syntax brevity
        x_train = self.x_train
        x_test = self.x_test
        y_train = self.y_train
        y_test = self.y_test

        # construct, run, and save MLP model
        # https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
        model = Sequential([
            Dense(neurons, input_dim=x_train.shape[1], activation=activation),
            Dropout(dropout),
            Dense(4, activation="softmax")
        ])
        model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                  callbacks=[ModelCheckpoint("mlp_njustice.hdf5", monitor="val_loss", save_best_only=True)])

        # store MLP model
        self.model = model

    # purpose: score MLP model
    # effect: prints model scores for metric used, Cohen-Kappa, F1, and the Cohen-Kappa + F1 average
    def score_model(self):
        # copy member properties for syntax brevity
        model = self.model
        x_test = self.x_test
        y_test = self.y_test

        # calculate metric score
        ac = 100 * model.evaluate(x_test, y_test)[1]
        # calculate Cohen-Kappa score
        ck = cohen_kappa_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1))
        # calculate F1 score
        f1 = f1_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1), average="macro")

        # calculate Cohen-Kappa and F1 average
        mean = (ck + f1) / 2

        # print model scores
        print("Accuracy:", ac)
        print("Cohen Kappa:", ck)
        print("F1 score:", f1)
        print("Mean:", mean)

        # store model average score
        self.score = mean


# initialize parameter values for best MLP model
RESIZE = (105, 150)
APPLY_PADDING = False
APPLY_FILTER = False
APPLY_GRAYSCALE = False

MINMAX = False
SAMPLER = "over"

NEURONS = 1000
ACTIVATION = "relu"
DROPOUT = 0.2
OPTIMIZER = SGD(0.0001)
LOSS = "categorical_crossentropy"
METRICS = "accuracy"
# https://medium.com/mini-distill/effect-of-batch-size-on-training-dynamics-21c14f7a716e
BATCH_SIZE = 50
EPOCHS = 100

# construct, train, and run MLP model
mlp0 = MLP()
mlp0.load_data(RESIZE, APPLY_PADDING, APPLY_FILTER, APPLY_GRAYSCALE)
mlp0.prep_data(MINMAX, SAMPLER)
mlp0.train_model(NEURONS, ACTIVATION, DROPOUT, OPTIMIZER, LOSS, METRICS, BATCH_SIZE, EPOCHS)
mlp0.score_model()
