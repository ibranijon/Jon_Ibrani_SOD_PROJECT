import tensorflow as tf 
from tensorflow import keras 


def sod_model(input_shape=(128,128,3)):
    
    input_l = keras.Input(shape=input_shape)

    #Encoder - Conv2 and Maxpooling 128-64-32-16
    
    l = keras.layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu')(input_l)
    l = keras.layers.MaxPooling2D(pool_size=2)(l)

    l = keras.layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu')(l)
    l = keras.layers.MaxPooling2D(pool_size=2)(l)

    l = keras.layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu')(l)
    l = keras.layers.MaxPooling2D(pool_size=2)(l)

    #Decoder - Convtransport2d

    l = keras.layers.Conv2DTranspose(filters=64,kernel_size=3,strides=2,padding='same',activation='relu')(l)
    l = keras.layers.Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same',activation='relu')(l)
    l = keras.layers.Conv2DTranspose(filters=16,kernel_size=3,strides=2,padding='same',activation='relu')(l)
    
    output_l = keras.layers.Conv2D(filters=1, kernel_size=1 , padding='same',activation='sigmoid')(l)


    #Model 
    model = keras.Model(inputs = input_l,output = output_l)
    return model



def iou():
    pass

def loss():
    pass

def optimizer():
    pass


