"""A script for preprocessing images to be used by the program.
Creates 2048x2048 crops from images that are larger. This size was
chosen because it seems to be big enough so that enough relevant
context information is preserved.
"""

import cv2
import os

dirname = "dispscrop/"
target_size = 2048


def halfres(imgs):
    res = imgs[0].shape[0]
    res2 = res//2
    result = []
    for ii in imgs:
        result.extend([ii[:res2, :res2, :], ii[res2:, :res2, :],
                       ii[:res2, res2:, :], ii[res2:, res2:, :]])
    return result


files = os.listdir(dirname)
names = []
for ff in files:
    names.append(dirname + ff)

for imgname in names:
    img_orig = cv2.imread(imgname)
    imgs = [img_orig]
    res = img_orig.shape[0]
    while res > target_size:
        imgs = halfres(imgs)
        res = imgs[0].shape[0]

    for num, ii in enumerate(imgs):
        fullname = imgname[:-4]
        fullname += str(num)
        fullname += imgname[-4:]
        cv2.imwrite(fullname, ii)
    os.remove(imgname)
