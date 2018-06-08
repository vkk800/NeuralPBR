"""A script for preprocessing the texture images so that
they can be used by the program. It scales all the images in
the given directory to the closest power of two size.
"""

import cv2
import os

dirname = "disps_orig/"

names = os.listdir(dirname)

twopows = [512, 1024, 2048, 4096, 8192, 16384]

for nn in names:
    rewrite = False
    fullname = dirname+nn
    img = cv2.imread(fullname)
    resx, resy = img.shape[:2]
    minres = min(resx, resy)
    if resx != resy:
        img = img[:minres, :minres, :]
        rewrite = True
    if minres not in twopows:
        newres = twopows[next(index for index, twopow in enumerate(
            twopows) if minres < twopow)-1]
        img = cv2.resize(img, (newres, newres))
        rewrite = True

    if rewrite:
        cv2.imwrite(fullname, img)
