import re
from collections import deque
import random
import cv2
import os


class BatchCreator:
    """The batch creator outputs training examples by cropping samples of desired
    size from the full sized textures. Since loading large sized textures into memory
    can become a bottleneck for performance (and also because single image can be used for
    multiple training samples), the class also allows the user to control
    how often new images are loaded and how many are kept in the memory at the same time.

    # Arguments
        dir_x: String. Path for source textures.
        dir_y: String. Path for target textures (roughness, normal, etc.).
        max_open: Int. How many images to keep loaded at the same time.
        resin: Tuple of two ints. Resolution of cropped inputs.
        resout: Tuple of two ints. Resolution of cropped outputs.
        batch_size: Int. Number of images in a batch.
        p_load: Float between 0 and 1. The probability to load a new image on each new iteration.
        debug: Boolean. Print some helpful information while working.
        bwx: Boolean. Whether to convert the inputs to black-and-white.
        bwy: Boolean. Whether to convert the outputs to black-and-whie.
    """

    def __init__(self, dir_x, dir_y, max_open=16, resin=(256, 256), resout=(256, 256),
                 batch_size=8, p_load=0.1, debug=False, bwx=False,
                 bwy=False):
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.resin = resin
        self.resout = resout
        self.max_open = max_open
        self.p_load = p_load
        self.bwx = bwx
        self.bwy = bwy
        self.batch_size = batch_size

        x_contents = os.listdir(dir_x)
        y_contents = os.listdir(dir_y)
        self.names = []
        for name in os.listdir(dir_x):
            yname = re.sub(r"\_col", "_disp", name)
            if yname in y_contents:
                self.names.append((self.dir_x + name,
                                   self.dir_y + yname))
        random.shuffle(self.names)

        self.loaded_imgs = deque(maxlen=max_open)
        for _ in range(max_open):
            self._loadNew()

    def _loadNew(self):
        pair = random.choice(self.names)

        try:
            imgx = cv2.imread(pair[0])
            imgx = 2*(imgx/256-0.5)
            if self.bwx:
                imgx = imgx[:, :, 0].reshape((*self.imgx.shape[:2], 1))
            imgy = cv2.imread(pair[1])
            imgy = 2*(imgy/256-0.5)
            if self.bwy:
                imgy = imgy[:, :, 0].reshape((*imgy.shape[:2], 1))
            print("Loaded new image: {}".format(pair[0]))
            self.loaded_imgs.append((imgx, imgy))
        except:
            print("Loading a new image failed. Continuing...")

    def preprocess(self, img):
        """Converts an image from the standard 0-255 integer representation
        to a float in the range (-1, 1).
        """
        return img/128.-1.

    def deprocess(self, img):
        """Inverse operation of `preprocess()`
        """
        return ((img+1.)*128).astype(int)

    def __iter__(self):
        """This class can be used as an iterator.
        """
        return self

    def __next__(self):
        if random.random() < self.p_load:
            self._loadNew()

        batchx = []
        batchy = []

        for _ in range(self.batch_size):
            pick = random.choice(self.loaded_imgs)
            indx = random.randint(0, pick[0].shape[0]-self.resin[0])
            indy = random.randint(0, pick[0].shape[1]-self.resin[1])

            diffx = (self.resin[0]-self.resout[0])//2
            diffy = (self.resin[1]-self.resout[1])//2

            batchx.append(pick[0][indx:(indx+self.resin[0]),
                                  indy:(indy+self.resin[1]), :])
            batchy.append(pick[1][(indx+diffx):(indx+diffx+self.resout[0]),
                                  (indy+diffy):(indy+diffy+self.resout[1]), :])

        return (batchx, batchy)
