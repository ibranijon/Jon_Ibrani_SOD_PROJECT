import tensorflow as tf 
from tensorflow import keras 
from keras import layers, Model, losses, optimizers

def sod_model(input_shape=(128,128,3)):
    
    input_l = keras.Input(shape=input_shape)

    #Encoder - Conv2 and Maxpooling 128-64-32-16
    
    l = layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu')(input_l)
    l = layers.MaxPooling2D(pool_size=2)(l)

    l = layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu')(l)
    l = layers.MaxPooling2D(pool_size=2)(l)

    l = layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu')(l)
    l = layers.MaxPooling2D(pool_size=2)(l)

    #Decoder - Convtransport2d

    l = layers.Conv2DTranspose(filters=64,kernel_size=3,strides=2,padding='same',activation='relu')(l)
    l = layers.Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same',activation='relu')(l)
    l = layers.Conv2DTranspose(filters=16,kernel_size=3,strides=2,padding='same',activation='relu')(l)
    
    output_l = layers.Conv2D(filters=1, kernel_size=1 , padding='same',activation='sigmoid')(l)


    #Model 
    model = Model(inputs = input_l,output = output_l)
    return model


bce = losses.binary_crossentropy()

def iou():
    pass

def loss():
    bce = 2
    iou = iou(y_pred,y)
    loss = bce + 0.5 * (1-iou)
    return loss

def optimizer():
    pass


