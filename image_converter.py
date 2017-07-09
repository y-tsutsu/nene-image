import cv2
import os
import os.path
import shutil
import re
import matplotlib.pyplot as plt
import seaborn


def rename():
    for r, ds, fs in os.walk('./nene_test'):
        for f in fs:
            result = re.search(r'.+_\d+\.', f)
            if result:
                os.remove(os.path.join(r, f))
            # result = re.search(r'(.+ \()(\d+)(\).+)', f)
            # nf = os.path.join(r, '{0}{1:03d}{2}'.format(
            #     result.group(1), int(result.group(2)), result.group(3)))
            # os.rename(os.path.join(r, f), nf)


def clip_image():
    for r, ds, fs in os.walk('./nene_data/test/ALMOND'):
        for f in fs:
            filename = os.path.join(r, f)
            img = cv2.imread(filename)
            height, width, channels = img.shape
            clp = cv2.resize(img, (1000, 1000))
            # clp = img[80:1080, 290:1290]

            # size = tuple([img.shape[1], img.shape[0]])
            # center = tuple([int(size[0] / 2), int(size[1] / 2)])
            # angle = float(x)
            # scale = 1.0
            # rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
            # clp = cv2.warpAffine(img, rotation_matrix,
            #                     size, flags=cv2.INTER_CUBIC)

            # f1, f2 = os.path.splitext(filename)

            plt.imshow(clp)
            plt.show()
            # cv2.imwrite(filename, clp)


def move():
    count = 0
    for r, ds, fs in os.walk('./nene_data/ALMOND'):
        for f in fs:
            filename = os.path.join(r, f)
            if count % 7 == 0:
                shutil.move(filename, os.path.join(r, 'test', f))
            count += 1


def main():
    rename()
    print('end')


if __name__ == '__main__':
    main()
