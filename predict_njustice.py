import os
import numpy as np
from PIL import Image
from keras.models import load_model

os.system("sudo pip install PIL")


def predict(x):
    images = []

    for i in x:
        image = Image.open(i)
        image = image.resize((105, 150), resample=Image.NEAREST)
        image = np.asarray(image)
        images.append(image)

    x = np.array(images)
    x = x.reshape(len(x), -1)
    x = x / 255

    model = load_model('mlp_njustice.hdf5')
    y = np.argmax(model.predict(x), axis=1)

    return y, model


# assert isinstance(y, type(np.array([1])))
# assert y.shape == (len(x),)
# assert np.unique(y).max() <= 3 and np.unique(y).min() >= 0

# predict(["data/secret/" + file for file in os.listdir("data/secret") if file.endswith(".png")])
