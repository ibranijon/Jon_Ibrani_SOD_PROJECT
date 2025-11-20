import os
import glob
import numpy as np
import cv2
from sklearn import model_selection
import tensorflow as tf 

height = 128
width = 128
batch_size = 16

def load_data():
    #grab path of data_loader 
    base = os.path.dirname(os.path.abspath(__file__))
    
    #grab specific path to image and mask folder
    img_path = os.path.join(base,'Dataset','Image','*')
    mask_path = os.path.join(base,'Dataset','Mask','*')

    #getting paths to all images and masks
    images = sorted(glob.glob(img_path))
    masks = sorted(glob.glob(mask_path))

    #pairing all of the masks and the images to their one another in a giant list
    pairs = list(zip(images,masks))
    return pairs


#Excpets 2 inputs and 2 outpus to be released
def data_preprocessing(image_path,mask_path):
    #image loader and bgr to rgb
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    #mask loader and grayscale applied?
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
   
    #Resizing of both
    image = cv2.resize(image,(height,width), interpolation= cv2.INTER_LINEAR)
    mask = cv2.resize(mask, (height,width), interpolation= cv2.INTER_NEAREST)

    #Normalization values from 255 to 0-1
    #astype to float32 for tensor compatibility IMPORTANT OTHER TO ENSURE BOTH ARE FLOAT32 OTHERWISE IT WONT WORK
    image = image.astype(np.float32)/255.0
    mask = mask.astype(np.float32)/255.0

    #adding channel value to mask to go in 3d so its accepted by tensor
    mask = np.expand_dims(mask,axis=-1)

    return image,mask

def augmentation(image,mask):
    # randomizer
    if tf.random.uniform(())>0.5:
        # flip on both image mask
        image = tf.image.flip_left_right(image)
        mask = tf.image.flip_left_right(mask)
   
   
    if tf.random.uniform(())>0.5:
        # crop on both image and mask
        image = tf.image.random_crop(image,[112,112,3])
        mask = tf.image.random_crop(mask,[112,112,1])
        # resizing before it crashes
        image = tf.image.resize(image,(height,width))
        mask = tf.image.resize(mask,(height,width),method='nearest')

    # light only on image
    if tf.random.uniform(())>0.5:
        image = tf.image.random_brightness(image, max_delta=0.1)

    # ensure float32 AFTER resizing
    image = tf.cast(image, tf.float32)
    mask = tf.cast(mask, tf.float32)

    # return image, mask
    return image,mask

def create_dataset():
    
    #execute load data and grab huge list
    pairs = load_data()

    preprocessed_images = []
    preprocessed_masks = []

    #execute the preprocessing
    for image_path,mask_path in pairs:
        image,mask = data_preprocessing(image_path,mask_path)
        preprocessed_images.append(image)
        preprocessed_masks.append(mask)

    #train,test,validation split
    train_images, temp_images, train_masks, temp_masks = model_selection.train_test_split(preprocessed_images, preprocessed_masks, test_size=0.2, random_state=67)
    test_images, validation_images, test_masks, validation_masks = model_selection.train_test_split(temp_images,temp_masks,test_size=0.5,random_state=67)

    #Dataset for train
    #create tf dataset when using slcing to create tf dataset you input a tuple otherwise it crashes
    train_ds = tf.data.Dataset.from_tensor_slices((train_images,train_masks))
    #augmentation notice how augmentation does not have (), bc of tensor flow magic
    train_ds = train_ds.map(augmentation, num_parallel_calls=tf.data.AUTOTUNE)
    #batch
    train_ds = train_ds.batch(batch_size)
    #prefetch
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

    #dataset tf fro test set
    #create tf dataset
    test_ds = tf.data.Dataset.from_tensor_slices((test_images,test_masks))    
    #batch
    test_ds = test_ds.batch(batch_size)
    #prefetch
    test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

    #dataset tf fro validation set
    #create tf dataset
    val_ds = tf.data.Dataset.from_tensor_slices((validation_images,validation_masks))
    #batch
    val_ds = val_ds.batch(batch_size)
    #prefetch
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    

    return train_ds, val_ds, test_ds
    

if __name__ == "__main__":

    train_ds, val_ds, test_ds = create_dataset()

    print(">>> DATA LOADER WORKING <<<")

    print("Train batches:")
    for images, masks in train_ds.take(1):
        print("images:", images.shape)
        print("masks:", masks.shape)

    print("\nValidation batches:")
    for images, masks in val_ds.take(1):
        print("images:", images.shape)
        print("masks:", masks.shape)

    print("\nTest batches:")
    for images, masks in test_ds.take(1):
        print("images:", images.shape)
        print("masks:", masks.shape)

