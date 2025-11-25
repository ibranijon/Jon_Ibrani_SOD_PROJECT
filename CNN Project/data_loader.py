import os
import glob
import numpy as np
import cv2
from sklearn import model_selection
import tensorflow as tf 
from tensorflow import image


height = 128
width = 128
batch_size = 16

def load_data():
    #Grabs the path of data loader file
    base = os.path.dirname(os.path.abspath(__file__))
    
    #Grabs the paths for image and mask folders with all their content
    img_path = os.path.join(base,'Dataset','Image','*')
    mask_path = os.path.join(base,'Dataset','Mask','*')

    #Grabs all paths of the images and masks invidividually and sorts into two groups
    images = sorted(glob.glob(img_path))
    masks = sorted(glob.glob(mask_path))

    #Pairs the respective image with its mask inside one giant list
    pairs = list(zip(images,masks))
    return pairs



def data_preprocessing(image_path,mask_path):


    #Loads the image and converts it to RGB
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)

    #Loads the mask and converts it to grayscale
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
   
    #Resizing of both image and mask 
    image = cv2.resize(image,(height,width), interpolation= cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (height,width), interpolation= cv2.INTER_NEAREST)

    #Normalization of the values from 0-255 to 0-1
    #Moreover, uses float32 to make sure there is no issues with tensorflow

    image = image.astype(np.float32)/255.0
    mask = mask.astype(np.float32)/255.0

    #Masks don't have channels, but we need to give them one to ensure compatibility with tensorflow
    mask = np.expand_dims(mask,axis=-1)

    return image,mask

def augmentation(image,mask):
   
    #Horizontal Flip
    if tf.random.uniform(())>0.5:
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)


    #Vertical flip
    if tf.random.uniform(())>0.5:
        image = tf.image.flip_up_down(image)
        mask = tf.image.flip_up_down(mask)

    #Crop
    if tf.random.uniform(())>0.5:
        scales = tf.random.uniform([],0.85,1.15)
        new_h = tf.cast(height * scales, tf.int32)
        new_w = tf.cast(width * scales, tf.int32)

        image = tf.image.resize(image, (new_h, new_w))
        mask = tf.image.resize(mask, (new_h,new_w),method='nearest')

        image = tf.image.resize_with_crop_or_pad(image, height, width)
        mask =  tf.image.resize_with_crop_or_pad(mask,height, width)


    #Change of brightness for image only
    if tf.random.uniform(())>0.5:
        image = tf.image.random_brightness(image, max_delta=0.1)

    #Change of contrast for images only 
    if tf.random.uniform(())>0.7:
        image = tf.image.random_contrast(image, 0.9, 1.1)

    #Change of saturation for images only
    if tf.random.uniform(())>0.7:
        image = tf.image.random_saturation(image, 0.9, 1.1)
    
    image = tf.clip_by_value(image, 0.0, 1.0)
    
    #Ensure the type is float32 after augmentation to prevent mismatch with non augmented data
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    return image,mask

def create_dataset():
    
    #Execute data loading, stores it into pairs
    pairs = load_data()


    #Execute preprocessing for all images and masks and stores them into two lists
    preprocessed_images = []
    preprocessed_masks = []

    for image_path,mask_path in pairs:
        image,mask = data_preprocessing(image_path,mask_path)
        preprocessed_images.append(image)
        preprocessed_masks.append(mask)

    #Splitting of data into train, test and validation sets
    train_images, temp_images, train_masks, temp_masks = model_selection.train_test_split(preprocessed_images, preprocessed_masks, test_size=0.3, random_state=67)
    test_images, validation_images, test_masks, validation_masks = model_selection.train_test_split(temp_images,temp_masks,test_size=0.5,random_state=67)

    #Datapipelining 
    #Creation of train_ds. Application of augmentation, batching and prefetching
    train_ds = tf.data.Dataset.from_tensor_slices((train_images,train_masks))
    train_ds = train_ds.shuffle(2048)
    train_ds = train_ds.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

  
    #Creation of test_ds. Application of batching and prefetching
    test_ds = tf.data.Dataset.from_tensor_slices((test_images,test_masks))    
    test_ds = test_ds.batch(batch_size)
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    #Creation of val_ds. Application of batching and prefetching
    val_ds = tf.data.Dataset.from_tensor_slices((validation_images,validation_masks))
    val_ds = val_ds.batch(batch_size)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    

    return train_ds, val_ds, test_ds
    
