import chainer
import chainer.functions as F
import chainer.links as L


class CNN(chainer.Chain):
    CLASSES = 36

    def __init__(self, train=True):
        super(CNN, self).__init__(
            conv1=L.Convolution2D(1, 32, 5),
            conv2=L.Convolution2D(32, 64, 5),
            l1=L.Linear(1024, CNN.CLASSES)
        )
        self.train = train

    def __call__(self, x):
        h = F.max_pooling_2d(F.relu(self.conv1(x)), 2)
        h = F.max_pooling_2d(F.relu(self.conv2(h)), 2)
        return self.l1(h)
