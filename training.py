from network import CNN
from imageview import view_image
import chainer
import chainer.function as F
import chainer.links as L
from chainer import training
from chainer import serializers
from chainer.datasets import tuple_dataset
from chainer.training import extensions
import os
import os.path
import numpy as np
import skimage.io
import skimage.color
from skimage.transform import rescale


def load_data(dirname, filename):
    with open(os.path.join(dirname, filename), encoding='utf-8') as f:
        lines = f.readlines()

    IMAGE_SIZE = 48
    IN_CHANNELS = 3
    count = len(lines)
    xs = np.zeros((count, IN_CHANNELS, IMAGE_SIZE,
                   IMAGE_SIZE)).astype(np.float32)
    ys = np.zeros(count).astype(np.int32)

    for i, line in enumerate(lines):
        png, label = line.split()
        img = skimage.io.imread(os.path.join(dirname, 'image', png))
        if img.shape[2] == 4:
            img = skimage.color.rgba2rgb(img)
        height, width = img.shape[:2]
        img = rescale(img, (IMAGE_SIZE / height,
                            IMAGE_SIZE / width), mode='constant')
        im = img.astype(np.float32).reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
        xs[i, :, :, :] = im.transpose(0, 3, 1, 2)
        ys[i] = label

    return tuple_dataset.TupleDataset(xs, ys)


def main():
    model = L.Classifier(CNN())

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train = load_data('./data', 'train.txt')
    test = load_data('./data', 'test.txt')
    train_iter = chainer.iterators.SerialIterator(train, batch_size=100)
    test_iter = chainer.iterators.SerialIterator(
        test, batch_size=100, repeat=False, shuffle=False)

    updater = training.StandardUpdater(train_iter, optimizer, device=None)
    trainer = training.Trainer(updater, (100, 'epoch'), out='result')
    trainer.extend(extensions.Evaluator(test_iter, model, device=None))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss', 'main/accuracy', 'validation/main/accuracy']))
    trainer.extend(extensions.ProgressBar())

    trainer.run()

    modeldir = './model'
    if not os.path.isdir(modeldir):
        os.mkdir(modeldir)
    serializers.save_npz(os.path.join(modeldir, 'model.npz'), model)
    serializers.save_npz(os.path.join(modeldir, 'optimizer.npz'), optimizer)


if __name__ == '__main__':
    main()
