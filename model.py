import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def REBNCONV(x, filters, dilation_rate=1, name_prefix="rebn"):
    x = layers.Conv2D(filters, 3, padding='same', dilation_rate=dilation_rate,
                      kernel_initializer='he_normal', name=f"{name_prefix}_conv")(x)
    x = layers.BatchNormalization(name=f"{name_prefix}_bn")(x)
    x = layers.ReLU(name=f"{name_prefix}_relu")(x)
    return x

def RSU4(x_input, in_filters, mid_filters, out_filters, name_prefix="rsu4"):
    hxin = REBNCONV(x_input, out_filters, name_prefix=f"{name_prefix}_in")

    hx1 = REBNCONV(hxin, mid_filters, name_prefix=f"{name_prefix}_conv1")
    pool1 = layers.MaxPooling2D(pool_size=2)(hx1)

    hx2 = REBNCONV(pool1, mid_filters, name_prefix=f"{name_prefix}_conv2")
    pool2 = layers.MaxPooling2D(pool_size=2)(hx2)

    hx3 = REBNCONV(pool2, mid_filters, name_prefix=f"{name_prefix}_conv3")
    hx4 = REBNCONV(hx3, mid_filters, dilation_rate=2, name_prefix=f"{name_prefix}_conv4")

    hx3d = REBNCONV(layers.Concatenate()([hx4, hx3]), mid_filters, name_prefix=f"{name_prefix}_conv3d")
    hx3d_up = layers.UpSampling2D(size=2, interpolation='bilinear')(hx3d)

    hx2d = REBNCONV(layers.Concatenate()([hx3d_up, hx2]), mid_filters, name_prefix=f"{name_prefix}_conv2d")
    hx2d_up = layers.UpSampling2D(size=2, interpolation='bilinear')(hx2d)

    hx1d = REBNCONV(layers.Concatenate()([hx2d_up, hx1]), out_filters, name_prefix=f"{name_prefix}_conv1d")

    return layers.Add(name=f"{name_prefix}_add")([hx1d, hxin])

def Generator(input_shape=(512, 512, 1)):
    inputs = keras.Input(shape=input_shape)

    stage1 = RSU4(inputs, 1, 16, 64, name_prefix="stage1")
    pool12 = layers.MaxPooling2D(pool_size=2)(stage1)

    stage2 = RSU4(pool12, 64, 16, 64, name_prefix="stage2")
    pool23 = layers.MaxPooling2D(pool_size=2)(stage2)

    stage3 = RSU4(pool23, 64, 16, 64, name_prefix="stage3")
    pool34 = layers.MaxPooling2D(pool_size=2)(stage3)

    stage4 = RSU4(pool34, 64, 16, 64, name_prefix="stage4")

    stage3d = RSU4(layers.Concatenate()([
        layers.UpSampling2D(size=2, interpolation='bilinear')(stage4), stage3
    ]), 128, 16, 64, name_prefix="stage3d")

    stage2d = RSU4(layers.Concatenate()([
        layers.UpSampling2D(size=2, interpolation='bilinear')(stage3d), stage2
    ]), 128, 16, 64, name_prefix="stage2d")

    stage1d = RSU4(layers.Concatenate()([
        layers.UpSampling2D(size=2, interpolation='bilinear')(stage2d), stage1
    ]), 128, 16, 64, name_prefix="stage1d")

    output = layers.Conv2D(1, 1, padding='same', activation='sigmoid', name="final_output")(stage1d)

    return keras.Model(inputs, output, name="U2NETP_Gray2Gray")