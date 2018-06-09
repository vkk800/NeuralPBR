"""Trains Simplenet (see `model.py` documentation) to compute displacement maps
and saves some example results along with the weights after training.

Simplenet lacks any context awareness and thus the results are not very good.
This is more of a proof-of-concept to show that my training approach produces
anything meaningful at all. 
"""

import numpy as np
from neuralpbr.model import *
import keras as K
import neuralpbr.batch_creator as bc
import cv2

inshape = (256, 256)
weight_file = 'weights/weights_simple_disps.h5'

simplenet = create_simplenet(size_in=inshape, chans_in=3, 
                             chans_out=1, nfeats=32)

outshape = tuple([num.value for num in simplenet.output.shape.dims[1:]])
generator = bc.BatchCreator('data/colscrop/',
                            'data/dispscrop/',
                            max_open=12, resin=inshape, resout=outshape,
                            batch_size=16, p_load=0.5, debug=False,
                            bwy=True)

steps = 5
simplenet.compile(keras.optimizers.SGD(lr=0.05, momentum=0.5,
                                       decay=0.95), loss=keras.losses.mean_squared_error)
try:
    simplenet.load_weights(weight_file)
except:
    pass

for i in range(steps):
    print("Step:", i)
    x, y = next(generator)
    loss = simplenet.train_on_batch(np.asarray(x), np.asarray(y))
    print("Loss was:", loss)
simplenet.save_weights(weight_file)

# Visually check the accuracy with larger images.
inshape_big = (1024, 1024)
new_simplenet = create_simplenet(size_in=inshape_big, chans_in=3, 
                                 chans_out=1, nfeats=32)

new_simplenet.compile(keras.optimizers.SGD(lr=0.1, momentum=0.5),
                      loss=keras.losses.mean_squared_error)
new_simplenet.load_weights(weight_file)

generator.resin = inshape_big
generator.resout = outshape = tuple([num.value for num in new_simplenet.output.shape.dims[1:]])
generator.batch_size = 1

# Compute one sample per batch to conserve GPU memory.
nsamples = 8
for i in range(nsamples):
    x, y = next(generator)
    preds = new_simplenet.predict(np.asarray(x))
    ytrue = [generator.deprocess(img) for img in y]
    xtrue = [generator.deprocess(img) for img in x]
    ypredtrue = [generator.deprocess(img) for img in preds]
    cv2.imwrite("samples/src"+str(i)+".png", xtrue[0])
    cv2.imwrite("samples/true"+str(i)+".png", ytrue[0])
    cv2.imwrite("samples/pred"+str(i)+".png", ypredtrue[0])
