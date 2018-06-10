"""Trains autoencoder (see `model.py` documentation) to compute displacement maps
and saves some example results along with the weights after training.

The idea is similar to denoising autoencoders. Initial testing shows that the
results are not very good. More fiddling with the model is required.
"""

import numpy as np
from neuralpbr.model import *
import keras as K
import neuralpbr.batch_creator as bc
import cv2

inshape = (256, 256)
weight_file = 'weights/weights_autoenc_disps.h5'

encodernet = create_autoencoder(size_in=inshape, chans_in=3, 
                                chans_out=1, nfeats=32, depth=4)

outshape = tuple([num.value for num in encodernet.output.shape.dims[1:]])
generator = bc.BatchCreator('data/colscrop/',
                            'data/dispscrop/',
                            max_open=12, resin=inshape, resout=outshape,
                            batch_size=8, p_load=0.0, debug=False,
                            bwy=False)

steps = 1000
encodernet.compile(keras.optimizers.SGD(lr=0.1, momentum=0.5), loss=keras.losses.mean_squared_error)
try:
    encodernet.load_weights(weight_file)
except:
    pass

for i in range(steps):
    print("Step:", i)
    x, y = next(generator)
    loss = encodernet.train_on_batch(np.asarray(x), np.asarray(y))
    print("Loss was:", loss)
encodernet.save_weights(weight_file)

# Visually check the accuracy with larger images.
inshape_big = (1024, 1024)
new_encodernet = create_autoencoder(size_in=inshape_big, chans_in=3, 
                                    chans_out=1, nfeats=32, depth=4)

new_encodernet.compile(keras.optimizers.SGD(lr=0.1, momentum=0.5),
                      loss=keras.losses.mean_squared_error)
new_encodernet.load_weights(weight_file)

generator.resin = inshape_big
generator.resout = outshape = tuple([num.value for num in new_encodernet.output.shape.dims[1:]])
generator.batch_size = 1

# Compute one sample per batch to conserve GPU memory.
nsamples = 8
for i in range(nsamples):
    x, y = next(generator)
    preds = new_encodernet.predict(np.asarray(x))
    ytrue = [generator.deprocess(img) for img in y]
    xtrue = [generator.deprocess(img) for img in x]
    ypredtrue = [generator.deprocess(img) for img in preds]
    cv2.imwrite("samples/src"+str(i)+".png", xtrue[0])
    cv2.imwrite("samples/true"+str(i)+".png", ytrue[0])
    cv2.imwrite("samples/pred"+str(i)+".png", ypredtrue[0])
