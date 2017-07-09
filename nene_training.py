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


def load_data(dirname):
    dirs = ['ALMOND', 'PRETZ', 'SOURS']

    IMAGE_SIZE = 64
    IN_CHANNELS = 3
    count = 0
    for i, dir in enumerate(dirs):
        for r, ds, fs in os.walk(os.path.join(dirname, dir)):
            count += len(fs)
    xs = np.zeros((count, IN_CHANNELS, IMAGE_SIZE,
                   IMAGE_SIZE)).astype(np.float32)
    ys = np.zeros(count).astype(np.int32)

    idx = 0
    for i, dir in enumerate(dirs):
        for r, ds, fs in os.walk(os.path.join(dirname, dir)):
            for f in fs:
                filename = os.path.join(r, f)
                img = skimage.io.imread(filename)
                if img.shape[2] == 4:
                    img = skimage.color.rgba2rgb(img)
                height, width = img.shape[:2]
                img = rescale(img, (IMAGE_SIZE / height,
                                    IMAGE_SIZE / width), mode='constant')
                im = img.astype(np.float32).reshape(
                    1, IMAGE_SIZE, IMAGE_SIZE, 3)
                xs[idx, :, :, :] = im.transpose(0, 3, 1, 2)
                ys[idx] = i
                idx += 1

    return tuple_dataset.TupleDataset(xs, ys)


def main():
    model = L.Classifier(CNN())

    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    train = load_data('./nene_data/train')
    test = load_data('./nene_data/test')
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
