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
from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint


class MLP:
    def __init__(self):
        self.images = []
        self.labels = []

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

        self.model = None
        self.score = None

    def load_data(self, resize, apply_padding, apply_filter):
        image_files = [file for file in os.listdir("data/train") if file.endswith(".png")]

        for i in image_files:
            image = Image.open("data/train/" + i)

            if apply_padding:
                ratio = float(resize) / max(image.size)
                size = tuple([int(x * ratio) for x in image.size])
                temp_image = image.resize(size, resample=Image.NEAREST)
                image = Image.new("RGB", (resize, resize), (255, 255, 255))
                image.paste(temp_image, ((resize - size[0]) // 2, (resize - size[1]) // 2))
            else:
                image = image.resize(resize, resample=Image.NEAREST)

            if apply_filter:
                image = image.filter(ImageFilter.BLUR)

            image = np.asarray(image)
            self.images.append(image)

        text_files = [file for file in os.listdir("data/train") if file.endswith(".txt")]

        for i in text_files:
            label = open("data/train/" + i, "r")
            label = label.read()
            self.labels.append(label)

    def prep_data(self, minmax, sampler):
        x = np.array(self.images)
        y = np.array(self.labels)

        x = x.reshape(len(x), -1)

        if minmax:
            scaler = MinMaxScaler()
            x = scaler.fit_transform(x)
        else:
            x = x / 255

        if sampler == "over":
            ros = RandomOverSampler(random_state=0)
            x, y = ros.fit_sample(x, y)
        elif sampler == "under":
            rus = RandomUnderSampler(random_state=0)
            x, y = rus.fit_resample(x, y)

        le = LabelEncoder()
        y = le.fit_transform(y)
        y = to_categorical(y, num_classes=4)

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.30, stratify=y)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self, neurons, activation, dropout, optimizer, loss, metrics, batch_size, epochs):
        x_train = self.x_train
        x_test = self.x_test
        y_train = self.y_train
        y_test = self.y_test

        model = Sequential([
            Dense(neurons, input_dim=x_train.shape[1], activation=activation),
            Dropout(dropout),
            Dense(4, activation="softmax")
        ])
        model.compile(optimizer=optimizer, loss=loss, metrics=[metrics])
        model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_data=(x_test, y_test),
                  callbacks=[ModelCheckpoint("mlp_njustice.hdf5", monitor="val_loss", save_best_only=True)])

        self.model = model

    def score_model(self):
        model = self.model
        x_test = self.x_test
        y_test = self.y_test

        ac = 100 * model.evaluate(x_test, y_test)[1]
        ck = cohen_kappa_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1))
        f1 = f1_score(np.argmax(model.predict(x_test), axis=1), np.argmax(y_test, axis=1), average="macro")

        mean = (ck + f1) / 2

        print("Accuracy:", ac)
        print("Cohen Kappa:", ck)
        print("F1 score:", f1)
        print("Mean:", mean)

        self.score = mean


RESIZE = (105, 150)
APPLY_PADDING = False
APPLY_FILTER = False

MINMAX = False
SAMPLER = "over"

NEURONS = 550
ACTIVATION = "relu"
DROPOUT = 0.2
OPTIMIZER = SGD(0.001)
LOSS = "categorical_crossentropy"
METRICS = "accuracy"
BATCH_SIZE = 200
EPOCHS = 5

scores = []

mlp0 = MLP()
mlp0.load_data(RESIZE, APPLY_PADDING, APPLY_FILTER)
mlp0.prep_data(MINMAX, SAMPLER)
mlp0.train_model(NEURONS, ACTIVATION, DROPOUT, OPTIMIZER, LOSS, METRICS, BATCH_SIZE, EPOCHS)
mlp0.score_model()
scores.append(mlp0.score)

mlp1 = MLP()
mlp1.load_data(RESIZE, APPLY_PADDING, APPLY_FILTER)
mlp1.prep_data(MINMAX, SAMPLER)
mlp1.train_model(NEURONS, ACTIVATION, DROPOUT, Adam(), LOSS, METRICS, BATCH_SIZE, EPOCHS)
mlp1.score_model()
scores.append(mlp1.score)

mlp2 = MLP()
mlp2.load_data(RESIZE, APPLY_PADDING, APPLY_FILTER)
mlp2.prep_data(MINMAX, SAMPLER)
mlp2.train_model(NEURONS, ACTIVATION, DROPOUT, OPTIMIZER, "kullback_leibler_divergence", METRICS, BATCH_SIZE, EPOCHS)
mlp2.score_model()
scores.append(mlp2.score)

mlp3 = MLP()
mlp3.load_data(RESIZE, APPLY_PADDING, APPLY_FILTER)
mlp3.prep_data(MINMAX, SAMPLER)
mlp3.train_model(NEURONS, ACTIVATION, DROPOUT, Adam(), "kullback_leibler_divergence", METRICS, BATCH_SIZE, EPOCHS)
mlp3.score_model()
scores.append(mlp3.score)

print(scores)

# https://github.com/amir-jafari/Deep-Learning/tree/master/Exam_MiniProjects
# https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
