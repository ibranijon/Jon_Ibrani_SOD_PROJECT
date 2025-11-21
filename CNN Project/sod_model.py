import tensorflow as tf 
from tensorflow import keras 
from keras import layers, Model, losses, optimizers,Input

def sod_model(input_shape=(128,128,3)):
    
    input_l = Input(shape=input_shape)

    #Encoder - Conv2 and Maxpooling 16-32-64-128
    
    l = layers.Conv2D(filters=32,kernel_size=3,padding='same',activation='relu')(input_l)
    l = layers.MaxPooling2D(pool_size=2)(l)

    l = layers.Conv2D(filters=64,kernel_size=3,padding='same',activation='relu')(l)
    l = layers.MaxPooling2D(pool_size=2)(l)

    l = layers.Conv2D(filters=128,kernel_size=3,padding='same',activation='relu')(l)
    l = layers.MaxPooling2D(pool_size=2)(l)

    #Decoder - Convtransport2d 128-64-32-16

    l = layers.Conv2DTranspose(filters=64,kernel_size=3,strides=2,padding='same',activation='relu')(l)
    l = layers.Conv2DTranspose(filters=32,kernel_size=3,strides=2,padding='same',activation='relu')(l)
    l = layers.Conv2DTranspose(filters=16,kernel_size=3,strides=2,padding='same',activation='relu')(l)
    
    output_l = layers.Conv2D(filters=1, kernel_size=1 , padding='same',activation='sigmoid')(l)


    #Model 
    model = Model(inputs = input_l,outputs = output_l)
    return model


bce_fn = losses.BinaryCrossentropy()

#IoU for Training
def soft_IoU(y_true,y_pred,epsilon=1e-7):

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true)+ tf.reduce_sum(y_pred) - intersection
    iou = (intersection)/(union + epsilon)
    return iou

#IoU for Evaluation
def hard_IoU(y_true,y_pred,epsilon=1e-7):

    #Casting from probability to hard 0 or 1 
    y_defined = tf.cast(y_pred>0.5, dtype=tf.float32)

    #IoU Caculation
    intersection = tf.reduce_sum(y_true * y_defined)
    union = tf.reduce_sum(y_true)+ tf.reduce_sum(y_defined) - intersection
    iou = (intersection)/(union + epsilon)
    return iou

#Loss Function combining both bce and IoU
def sod_loss(y_true, y_pred):
    bce = bce_fn(y_true, y_pred)
    iou = soft_IoU(y_true, y_pred)
    loss = bce + 0.5 * (1-iou)
    return loss

def optimizer():
    return optimizers.Adam(learning_rate=0.001)


