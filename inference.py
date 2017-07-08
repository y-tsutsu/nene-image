import string
import skimage.io
import skimage.color
from skimage.transform import rescale
import os.path
from imageview import view_image
from network import CNN
import chainer.links as L
from chainer import serializers
from chainer import Variable
import numpy as np
import matplotlib.pyplot as plt
import seaborn


def inference(img):
    model = L.Classifier(CNN())
    serializers.load_npz('./model/model.npz', model)

    img = skimage.color.rgb2gray(img)
    height, width = img.shape[:2]
    IMAGE_SIZE = 28
    img = rescale(img, (IMAGE_SIZE / height,
                        IMAGE_SIZE / width), mode='constant')
    im = img.astype(np.float32).reshape(1, 1, IMAGE_SIZE, IMAGE_SIZE)
    x = Variable(im)
    y = model.predictor(x)
    [pred] = y.data
    recog = np.argmax(pred)
    return recog, im.reshape(IMAGE_SIZE, IMAGE_SIZE)


def main():
    labels = string.digits + string.ascii_uppercase
    for i, label in enumerate(labels):
        img = skimage.io.imread(os.path.join('./sample', '{}.png'.format(i)))
        recog, img_gray = inference(img)
        plt.subplot(4, 10, i + 1)
        plt.title(labels[recog], size=12, color='k' if i == recog else 'r')
        plt.axis([0, 28, 28, 0])
        plt.imshow(img_gray, cmap='gray')
        plt.tick_params(labelbottom='off')
        plt.tick_params(labelleft='off')
    plt.show()


if __name__ == '__main__':
    main()
