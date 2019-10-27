import os
import numpy as np
from PIL import Image
# from PIL import ImageFilter
# from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
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

    def load_data(self):
        image_files = [file for file in os.listdir("data/train") if file.endswith(".png")]

        for i in image_files:
            image = Image.open("data/train/" + i)
            image = image.resize((105, 150), resample=Image.NEAREST)  # (87, 63) (123, 238)
            image = np.asarray(image)
            self.images.append(image)

        text_files = [file for file in os.listdir("data/train") if file.endswith(".txt")]

        for i in text_files:
            label = open("data/train/" + i, "r")
            label = label.read()
            self.labels.append(label)

    def prep_data(self):
        x = np.array(self.images)
        y = np.array(self.labels)

        # x = x.reshape(len(x), -1)
        # ros = RandomOverSampler(random_state=0)
        # x, y = ros.fit_sample(x, y)

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.30, stratify=y)

        x_train = x_train.reshape(len(x_train), -1)
        x_test = x_test.reshape(len(x_test), -1)
        x_train = x_train / 255
        x_test = x_test / 255

        le = LabelEncoder()
        y_train = le.fit_transform(y_train)
        y_test = le.fit_transform(y_test)

        y_train = to_categorical(y_train, num_classes=4)
        y_test = to_categorical(y_test, num_classes=4)

        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test

    def train_model(self):
        model = Sequential([
            Dense(512, input_dim=self.x_train.shape[1], activation="relu"),
            Dropout(0.2),
            Dense(512, activation="relu"),
            Dropout(0.2),
            Dense(4, activation="softmax")
        ])
        model.compile(optimizer=SGD(0.001), loss="categorical_crossentropy", metrics=["accuracy"])
        model.fit(self.x_train, self.y_train, batch_size=250, epochs=10, validation_data=(self.x_test, self.y_test),
                  callbacks=[ModelCheckpoint("mlp_njustice.hdf5", monitor="val_loss", save_best_only=True)])

        self.model = model

    def score_model(self):
        print("Accuracy:", 100 * self.model.evaluate(self.x_test, self.y_test)[1], "%")
        print("Cohen Kappa", cohen_kappa_score(np.argmax(self.model.predict(self.x_test), axis=1),
                                               np.argmax(self.y_test, axis=1)))
        print("F1 score", f1_score(np.argmax(self.model.predict(self.x_test), axis=1),
                                   np.argmax(self.y_test, axis=1), average="macro"))


mlp = MLP()
mlp.load_data()
mlp.prep_data()
mlp.train_model()
mlp.score_model()

# https://github.com/amir-jafari/Deep-Learning/tree/master/Exam_MiniProjects
# https://github.com/keras-team/keras/blob/master/examples/mnist_mlp.py
