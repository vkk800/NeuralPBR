import numpy as np
from model import *
import keras as K
import batch_creator as bc
import cv2

full_size = (1028, 1028)
out_size = (256, 256)

unet = create_cropped_unet(size_in=full_size, cropped_size=out_size,
                           chans_in=3, chans_out=1, nfeats=32, depth=3,
                           pool_factor=4, pooling='avg', cheap_convs=True)

generator = bc.BatchCreator('data/colscrop/',
                            'data/dispscrop/',
                            max_open=12, resin=full_size, resout=out_size,
                            batch_size=16, p_load=0.5, debug=False,
                            bwy = True)

steps = 1000

unet.compile(keras.optimizers.SGD(lr=0.05, momentum=0.5, decay=0.95), loss=keras.losses.mean_squared_error)

#unet.load_weights('uweights.hdf5')

for i in range(steps):
    print("Step:", i)
    x,y = generator.getBatch()
    loss = unet.train_on_batch(np.asarray(x), np.asarray(y))
    print("Loss was:", loss)

unet.save_weights('uweights.hdf5')

# Visually check the accuracy with larger outputs
out_size2 = (512, 512)
new_unet = create_cropped_unet(size_in=full_size, cropped_size=out_size2,
                               chans_in=3, chans_out=1, nfeats=32, depth=3,
                               pool_factor=4, pooling='avg', cheap_convs=True)
new_unet.compile(keras.optimizers.SGD(lr=0.1, momentum=0.5), loss=keras.losses.mean_squared_error)
new_unet.load_weights('uweights.hdf5')

generator.resin = full_size
generator.resout = out_size2
generator.batch_size=6

x,y = generator.getBatch()
preds = new_unet.predict(np.asarray(x))
ytrue = [generator.deprocess(img) for img in y]
xtrue = [generator.deprocess(img) for img in x]
ypredtrue = [generator.deprocess(img) for img in preds]

for i,img in enumerate(xtrue):
    cv2.imwrite("src"+str(i)+".png", img)
for i,img in enumerate(ytrue):
    cv2.imwrite("true"+str(i)+".png", img)
for i,img in enumerate(ypredtrue):
    cv2.imwrite("pred"+str(i)+".png", img)
