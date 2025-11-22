#imports
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from data_loader import create_dataset
from sod_model import set_sod_model, optimizer


#Model Prediction into the test set


def load_model_testset():

    sod_model = set_sod_model()
    opt = optimizer()
    
    checkp = tf.train.Checkpoint(model=sod_model,optimizer=opt,step=tf.Variable(0))
    manager = tf.train.CheckpointManager(checkp,'./checkpoints', max_to_keep=3)
    if manager.latest_checkpoint:
        print(f'Loading checkpoint {manager.latest_checkpoint}')
        checkp.restore(manager.latest_checkpoint).expect_partial()
    else:
        print("No checkpoint")

    __, __, test_set = create_dataset()
    return sod_model, test_set


#Metric Computer 


def compute_metrics(sod_model, test_set):
    precision_list = []
    recall_list = []
    f1_list = []
    mae_list = []
    iou_list = []
    eps = 1e-7 


    for images,masks in test_set:
        mask_preds = sod_model(images, training=False)


        #Binary Results of Prediction and Actual Data
        mask_preds = tf.cast((mask_preds>0.5), tf.float32)
        masks = tf.cast(masks,tf.float32)

        #Calculation of TP,FP,FN
        TP = tf.reduce_sum(tf.cast(((mask_preds==1) & (masks == 1)),tf.float32))
        FN = tf.reduce_sum(tf.cast(((mask_preds==0) & (masks == 1)),tf.float32))
        FP = tf.reduce_sum(tf.cast(((mask_preds==1) & (masks == 0)),tf.float32))
        
        #Calculation of metrics

        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        f1 = 2 * (precision * recall) / (recall + precision + eps)
        iou = TP / (TP + FP + FN + eps)
        mae = tf.reduce_mean(tf.abs(mask_preds-masks))
        
        #Filling information

        precision_list.append(precision.numpy())
        recall_list.append(recall.numpy())
        f1_list.append(f1.numpy())
        iou_list.append(iou.numpy())
        mae_list.append(mae.numpy())

    return precision_list, recall_list, f1_list, iou_list, mae_list

import matplotlib.pyplot as plt

import random

def visualize_predictions(model, test_set, num_images=3):
    # Collect all test images into lists (small, test set only)
    all_images = []
    all_masks = []

    for images, masks in test_set:
        for i in range(images.shape[0]):
            all_images.append(images[i])
            all_masks.append(masks[i])

    # Pick random indices
    indices = random.sample(range(len(all_images)), num_images)

    for idx in indices:
        img = all_images[idx].numpy()
        gt_mask = all_masks[idx].numpy().squeeze()

        pred = model(tf.expand_dims(all_images[idx], 0), training=False)
        pred_mask = tf.cast(pred > 0.5, tf.float32)[0].numpy().squeeze()

        # Overlay (in red)
        overlay = img.copy()
        overlay[..., 0] = np.maximum(overlay[..., 0], pred_mask)

        plt.figure(figsize=(12, 8))

        plt.subplot(1, 4, 1)
        plt.title("Original")
        plt.imshow(img)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Ground Truth")
        plt.imshow(gt_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Prediction")
        plt.imshow(pred_mask, cmap="gray")
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title("Overlay")
        plt.imshow(overlay)
        plt.axis("off")

        plt.show()



def evaluate():
    #Call of model, data and metric compiler
    sod_model, test_set = load_model_testset()
    precision_list, recall_list, f1_list, iou_list, mae_list = compute_metrics(sod_model,test_set)


    #Calculation of metric averages
    precision = sum(precision_list)/len(precision_list)
    recall = sum(recall_list)/len(recall_list)
    f1 = sum(f1_list)/len(f1_list)
    iou = sum(iou_list)/len(iou_list)
    mae = sum(mae_list)/len(mae_list)

    print(f'Precision:{precision}')
    print(f'Recall:{recall}')
    print(f'F1:{f1}')
    print(f'IOU:{iou}')
    print(f'Mean average error:{mae}')

    print("\nGenerating visualizations...\n")
    visualize_predictions(sod_model, test_set, num_images=3)



def main():
    evaluate()

if __name__ == "__main__":
    evaluate()



#gathering of the confusion matrix elemetns 
#return precision,recall,f1,iou and mae


#visualizer