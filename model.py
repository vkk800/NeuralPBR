import keras
from keras.layers import *
from keras.utils import plot_model

def convblock(input_level, feats):
    """A block that is four pixels smaller than the input.
    Used in the various models below.
    """
    out = Conv2D(feats, (1, 1), activation='relu', padding='valid')(input_level)
    out = Conv2D(feats, (3, 3), activation='relu', padding='valid')(out) 
    out = Conv2D(feats, (3, 3), activation='relu', padding='valid')(out)
    return out

def convblock_cheap(input_level, feats):
    """A block that is two pixels smaller than the input
    """
    out = Conv2D(feats, (1, 1), activation='relu', padding='valid')(input_level)
    out = Conv2D(feats, (3, 3), activation='relu', padding='valid')(out)
    return out


def get_crop(larger, smaller):
    """Computes the number of pixels that need to be cropped in order to make
    the two layers the same size.
    """
    dwidth = (larger.get_shape()[1]-smaller.get_shape()[1]).value
    dheight = (larger.get_shape()[2]-smaller.get_shape()[2]).value

    cropw = (dwidth//2, dwidth//2 + dwidth%2)
    croph = (dheight//2, dheight//2 + dheight%2)

    return (cropw, croph)
        
def create_net_simple(size_in=(64, 64), chans_in=3, chans_out=3, nfeats=32,
               depth=3, pooling='max'):
    """A simple model that just applies a bunch of convolutions without any context
    awareness.
    """
    
    if pooling == 'max':
        PoolingFun = MaxPooling2D
    elif pooling == 'avg':
        PoolingFun = AveragePooling2D
    else:
        raise AttributeError("Invalid pooling function")
    
    input_layer = Input(shape=(*size_in, chans_in))
    x = Conv2D(nfeats, (1,1), padding='valid', activation='relu')(input_layer)
    x = Conv2D(nfeats, (5,5), padding='valid', activation='relu')(x)
    x = Conv2D(nfeats, (5,5), padding='valid', activation='relu')(x)
    x = Conv2D(chans_out, (1,1), padding='valid', activation=None)(x)

    model = keras.Model(inputs=input_layer, outputs=x)
    
    return model

def create_unet(size_in=(512, 512), chans_in=3, chans_out=3, nfeats=32,
               depth=3, pooling='max'):
    """Unet style model with adjustable depth and number of features per
    layer.
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
        levelsdn.append(convblock(newlayer, nfeats*2**(i+1)))

    # The lowest level
    lowest = Conv2D(nfeats*2**(depth+1), (1,1), activation='relu')(levelsdn[-1])
    lowest = Conv2D(nfeats*2**(depth+1), (3,3), activation='relu', padding='valid')(lowest)
    lowest = Conv2D(nfeats*2**(depth+1), (3,3), activation='relu', padding='valid')(lowest)
    lowest = Conv2D(nfeats*2**(depth+1), (1,1), activation='relu')(lowest)

    # The upscaling branch
    levelsup = [lowest]
    for i in range(depth):
        upsampled = UpSampling2D(size=(2, 2))(levelsup[-1])
        cropped = Cropping2D(cropping=get_crop(levelsdn[-(i+2)], upsampled))(levelsdn[-(i+2)])
        conc = Concatenate()([upsampled, cropped])
        levelsup.append(convblock(conc, nfeats*2**(depth-i-1)))

    output = Conv2D(chans_out, (1,1), activation=None)(levelsup[-1])

    model = keras.Model(inputs=input_layer, outputs=output)
    return model

def create_unet_cheap(size_in=(512, 512), chans_in=3, chans_out=3, nfeats=32,
                     depth=2, pooling='avg'):
    """A cheaper variation of Unet.
    """

    if pooling == 'max':
        PoolingFun = MaxPooling2D
    elif pooling == 'avg':
        PoolingFun = AveragePooling2D
    else:
        raise AttributeError("Invalid pooling function")
    
    input_layer = Input(shape=(*size_in, chans_in))
    
    levelsdn = [convblock(input_layer, nfeats)]
    for i in range(depth):
        newlayer = PoolingFun(pool_size=4)(levelsdn[-1])
        levelsdn.append(convblock_cheap(newlayer, nfeats*2**(i+1)))

    lowest = Conv2D(nfeats*2**(depth+1), (1,1), activation='relu')(levelsdn[-1])
    lowest = Conv2D(nfeats*2**(depth+1), (3,3), activation='relu', padding='valid')(lowest)
    lowest = Conv2D(nfeats*2**(depth+1), (1,1), activation='relu')(lowest)

    levelsup = [lowest]        
    for i in range(depth):
        upsampled = UpSampling2D(size=(4, 4))(levelsup[-1])
        cropped = Cropping2D(cropping=get_crop(levelsdn[-(i+2)], upsampled))(levelsdn[-(i+2)])
        conc = Concatenate()([upsampled, cropped])
        levelsup.append(convblock_cheap(conc, nfeats*2**(depth-i-1)))

    output = Conv2D(chans_out, (1,1), activation=None)(levelsup[-1])

    model = keras.Model(inputs=input_layer, outputs=output)
    return model

def create_cropped_unet(size_in=(1024, 1024), cropped_size=(256, 256),
                        chans_in=3, chans_out=3, nfeats=32,
                        depth=2, pool_factor=4, pooling='avg',
                        cheap_convs=True):
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

    if cheap_convs:
        convfun = convblock_cheap
    else:
        convfun = convblock

    input_layer = Input(shape=(*size_in, chans_in))
    cropped_input = Cropping2D(cropping=((size_in[0]-cropped_size[0])//2-2,
                                         (size_in[1]-cropped_size[1])//2-2))(input_layer)

    # The downscaling branch
    levelsdn = [input_layer]
    for i in range(depth):
        newlayer = PoolingFun(pool_size=pool_factor)(levelsdn[-1])
        levelsdn.append(convfun(newlayer, nfeats*2**(i+1)))

    # The lowest level
    lowest = Conv2D(nfeats*2**(depth+1), (1, 1), activation='relu')(levelsdn[-1])
    lowest = Conv2D(nfeats*2**(depth+1), (3, 3), activation='relu', padding='valid')(lowest)
    lowest = Conv2D(nfeats*2**(depth+1), (1, 1), activation='relu')(lowest)

    # The upscaling branch
    levelsup = [lowest]
    for i in range(depth):
        upsampled = UpSampling2D(size=pool_factor)(levelsup[-1])
        cropped = Cropping2D(cropping=get_crop(levelsdn[-(i+2)], upsampled))(levelsdn[-(i+2)])
        total = Concatenate()([upsampled, cropped])
        levelsup.append(convfun(total, nfeats*2**(depth-i-1)))

    output_cropped = Cropping2D(cropping=get_crop(levelsup[-1], cropped_input))(levelsup[-1])
    output_cropped = Conv2D(nfeats, (3, 3), activation='relu', padding='valid')(output_cropped)
    output_cropped = Conv2D(nfeats, (3, 3), activation='relu', padding='valid')(output_cropped)
    output_cropped = Conv2D(nfeats, (1, 1), activation='relu')(output_cropped)
    output = Conv2D(chans_out, (1, 1), activation=None)(output_cropped)

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

    out = Conv2D(chans_out, (1,1), activation=None)(x0_2)

    model = keras.Model(inputs=input_layer, outputs=out)

    return model

   
 
