# install dependencies
import os
os.system("sudo pip install PIL")

# load dependencies
import numpy as np
from PIL import Image
from keras.models import load_model


# purpose: predict class labels on passed image data
# input: list of paths to images
# output: 1) collection of predicted class labels
#         2) Keras model instance
def predict(x):
    # initialize storage for images
    images = []

    # loop through image paths
    for i in x:
        # capture image in focus
        image = Image.open(i)
        # scale image in focus
        image = image.resize((105, 150), resample=Image.NEAREST)
        # convert image in focus to array and store it
        image = np.asarray(image)
        images.append(image)

    # convert image storage to array
    x = np.array(images)
    # flatten image array
    x = x.reshape(len(x), -1)
    # divide image arrays by 255 (converts RBG values into 0-1 range)
    x = x / 255

    # load model and run prediction
    model = load_model('mlp_njustice.hdf5')
    y = np.argmax(model.predict(x), axis=1)

    return y, model
