"""Models for transforming images into PBR texture maps.
At the moment, the models included are:
- "Simplenet": Just a few trainable convolutional filters stacked.
- Custom U-Net: U-Net inspired architecture that should have some context awareness.
- Cropped U-Net: Optimized version of the U-Net that can have context awareness of
    a larger region than is being transformed.
- Autoencoder type model.
- Plus some other half-assed experimental ideas.
Some of these are completely untested and the rest have only gone through
some preliminary sanity checks. More thorough testing is required.

TODO:
- Test the quality of the nets by training them on a beefier GPU.
- Do some hyperparameter optimization for all of the nets.
- Invent some new models. Check e.g. the literature on other image segmentation
    architectures (SegNet, DeepLab, dilated convnets, etc.).
"""

import keras
from keras.layers import *
from keras.utils import plot_model


def convblock(input_level, feats, nconvs=2):
    """A block that is four pixels smaller than the input.
    Used in the various models below.

    # Arguments
        input_level: Keras layer. Input for the block.
        feats: Int. Number of feature channels.
        nconvs: Int. How many 3x3 convolutions to apply.

    # Returns
        Keras layer after one 1x1 convolution and `nconvs` 3x3 convolutions
        have been applied to the input.
    """
    out = Conv2D(feats, (1, 1), activation='relu',
                 padding='valid')(input_level)
    for _ in range(nconvs):
        out = Conv2D(feats, (3, 3), activation='relu', padding='valid')(out)
    return out


def get_crop(larger, smaller):
    """Computes the number of pixels that need to be cropped in order to make
    the two layers the same size.

    # Arguments
        larger: Keras layer. The larger of the two inputs.
        smalle: Keras layer. The smaller of the two inputs.

    # Returns
        Tuple of two ints. The right cropping size for width and height.
    """
    dwidth = (larger.get_shape()[1]-smaller.get_shape()[1]).value
    dheight = (larger.get_shape()[2]-smaller.get_shape()[2]).value

    cropw = (dwidth//2, dwidth//2 + dwidth % 2)
    croph = (dheight//2, dheight//2 + dheight % 2)

    return (cropw, croph)


def create_simplenet(size_in=(64, 64), chans_in=3, chans_out=3,
                     nfeats=32, layers=2, **kwargs):
    """A simple model that just applies a bunch of convolutions without any context
    awareness.
    """
    input_layer = Input(shape=(*size_in, chans_in))

    x = Conv2D(nfeats, (1, 1), padding='valid', activation='relu')(input_layer)
    for _ in range(layers):
            x = Conv2D(nfeats, (5, 5), padding='valid', activation='relu')(x)
    x = Conv2D(chans_out, (1, 1), padding='valid', activation='sigmoid')(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    return model


def create_dilated_simplenet(size_in=(64, 64), chans_in=3, chans_out=3,
                             nfeats=32, layers=2, **kwargs):
    """A simple model that just applies a bunch of convolutions without any context
    awareness.
    """
    input_layer = Input(shape=(*size_in, chans_in))

    x = Conv2D(nfeats, (1, 1), padding='valid', activation='relu')(input_layer)
    x = Conv2D(nfeats, (3, 3), padding='valid', activation='relu')(x)
    for _ in range(layers):
        x = Conv2D(nfeats, (3, 3), padding='valid', activation='relu', dilation_rate=3)(x)
    x = Conv2D(nfeats, (3, 3), padding='valid', activation='relu')(x)
    x = Conv2D(chans_out, (1, 1), padding='valid', activation='sigmoid')(x)

    model = keras.Model(inputs=input_layer, outputs=x)

    return model


def create_unet(size_in=(512, 512), chans_in=3, chans_out=3, nfeats=32,
                depth=3, pooling='max', nconvs=1):
    """Unet style model with adjustable depth and number of features per
    layer. The difference to the "original Unet" is that the output is not
    a segmentation map, but an image `chans_out` number of color channels.
    """

    if pooling == 'max':
        PoolingFun = MaxPooling2D
    elif pooling == 'avg':
        PoolingFun = AveragePooling2D
    else:
        raise AttributeError("Invalid pooling function")

    input_layer = Input(shape=(*size_in, chans_in))

    # The downscaling branch
    levelsdn = [convblock(input_layer, nfeats)]
    for i in range(depth):
        newlayer = PoolingFun(pool_size=(2, 2))(levelsdn[-1])
        levelsdn.append(convblock(newlayer, nfeats*2**(i+1), nconvs=nconvs))

    # The lowest level
    lowest = Conv2D(nfeats*2**(depth+1), (1, 1),
                    activation='relu')(levelsdn[-1])
    for _ in range(nconvs):
        lowest = Conv2D(nfeats*2**(depth+1), (3, 3),
                        activation='relu', padding='valid')(lowest)
        lowest = Conv2D(nfeats*2**(depth+1), (3, 3),
                        activation='relu', padding='valid')(lowest)
    lowest = Conv2D(nfeats*2**(depth+1), (1, 1), activation='relu')(lowest)

    # The upscaling branch
    levelsup = [lowest]
    for i in range(depth):
        upsampled = UpSampling2D(size=(2, 2))(levelsup[-1])
        cropped = Cropping2D(cropping=get_crop(
            levelsdn[-(i+2)], upsampled))(levelsdn[-(i+2)])
        conc = Concatenate()([upsampled, cropped])
        levelsup.append(convblock(conc, nfeats*2**(depth-i-1), nconvs=nconvs))

    output = Conv2D(chans_out, (1, 1), activation='sigmoid')(levelsup[-1])
    model = keras.Model(inputs=input_layer, outputs=output)

    return model


def create_autoencoder(size_in=(512, 512), chans_in=3, chans_out=3,
                       nfeats=32, depth=2):
    """Autoencoder model.
    """

    input_layer = Input(shape=(*size_in, chans_in))
    x = input_layer
    
    for i in range(depth):
        x = Conv2D(nfeats*2**(depth-i-1), (3, 3), padding='same', strides=2)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = Conv2D(nfeats*2**(depth-i-1), (3, 3), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    for i in range(depth):
        x = Conv2D(nfeats*2**i, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(nfeats*2**i, (3, 3), activation='relu', padding='same')(x)
        x = UpSampling2D((2, 2))(x)

    output = Conv2D(chans_out, (3, 3), activation='sigmoid', padding='same')(x)
    model = keras.Model(input_layer, output)
    return model
    

def create_cropped_unet(size_in=(1024, 1024), cropped_size=(256, 256),
                        chans_in=3, chans_out=3, nfeats=32,
                        depth=2, pool_factor=4, pooling='avg',
                        nconvs=1):
    """An experimental variation of the unets above where the context
    information of the created map comes from a larger patch of the
    picture than the transformed image itself. This is a trick to
    make the model trainable on smaller GPU memory.
    """

    if pooling == 'max':
        PoolingFun = MaxPooling2D
    elif pooling == 'avg':
        PoolingFun = AveragePooling2D
    else:
        raise AttributeError("Invalid pooling function")

    input_layer = Input(shape=(*size_in, chans_in))
    cropped_input = Cropping2D(cropping=((size_in[0]-cropped_size[0])//2-2,
                                         (size_in[1]-cropped_size[1])//2-2))(input_layer)

    # The downscaling branch
    levelsdn = [input_layer]
    for i in range(depth):
        newlayer = PoolingFun(pool_size=pool_factor)(levelsdn[-1])
        levelsdn.append(convblock(newlayer, nfeats*2**(i+1), nconvs=nconvs))

    # The lowest level
    lowest = Conv2D(nfeats*2**(depth+1), (1, 1),
                    activation='relu')(levelsdn[-1])
    lowest = Conv2D(nfeats*2**(depth+1), (3, 3),
                    activation='relu', padding='valid')(lowest)
    lowest = Conv2D(nfeats*2**(depth+1), (1, 1), activation='relu')(lowest)

    # The upscaling branch
    levelsup = [lowest]
    for i in range(depth):
        upsampled = UpSampling2D(size=pool_factor)(levelsup[-1])
        cropped = Cropping2D(cropping=get_crop(
            levelsdn[-(i+2)], upsampled))(levelsdn[-(i+2)])
        total = Concatenate()([upsampled, cropped])
        levelsup.append(convblock(total, nfeats*2**(depth-i-1), nconvs=nconvs))

    output_cropped = Cropping2D(cropping=get_crop(
        levelsup[-1], cropped_input))(levelsup[-1])
    output_cropped = Conv2D(
        nfeats, (3, 3), activation='relu', padding='valid')(output_cropped)
    output_cropped = Conv2D(
        nfeats, (3, 3), activation='relu', padding='valid')(output_cropped)
    output_cropped = Conv2D(nfeats, (1, 1), activation='relu')(output_cropped)
    output = Conv2D(chans_out, (1, 1), activation='sigmoid')(output_cropped)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model


def model5(size_in=(512, 512), chans_in=3, chans_out=3, nfeats=32,
           depth=2, pooling='avg'):
    """I forgot what the point of this was.
    """

    if pooling == 'max':
        PoolingFun = MaxPooling2D
    elif pooling == 'avg':
        PoolingFun = AveragePooling2D
    else:
        raise AttributeError("Invalid pooling function")

    input_layer = Input(shape=(*size_in, chans_in))

    x0 = convblock_cheap(input_layer, nfeats)

    x1 = PoolingFun(4)(x0)
    x1 = convblock_cheap(x1, nfeats*2)

    x2 = PoolingFun(4)(x1)
    x2 = convblock_cheap(x2, nfeats*4)

    x3 = PoolingFun(4)(x2)
    x3 = convblock(x2, nfeats*8)

    up2 = UpSampling2D(size=4)(x3)

    crop2 = Cropping2D(cropping=get_crop(x2, up2))(x2)
    x2_2 = Concatenate()([up2, crop2])
    x2_2 = convblock_cheap(x2_2, nfeats*4)

    up1 = UpSampling2D(size=4)(x2_2)
    crop1 = Cropping2D(cropping=get_crop(x1, up1))(x1)
    x1_2 = Concatenate()([up1, crop1])
    x1_2 = convblock_cheap(x1_2, nfeats*2)

    up0 = UpSampling2D(size=4)(x1_2)
    crop0 = Cropping2D(cropping=get_crop(x0, up0))(x0)
    x0_2 = Concatenate()([up0, crop0])
    x0_2 = convblock_cheap(x0_1, nfeats)

    out = Conv2D(chans_out, (1, 1), activation='sigmoid')(x0_2)

    model = keras.Model(inputs=input_layer, outputs=out)

    return model
