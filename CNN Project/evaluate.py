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

        #Calculation of TP,FP,TN,FN
        TP = tf.reduce_sum(tf.cast(((mask_preds==1) & (masks == 1)),tf.float32))
        TN = tf.reduce_sum(tf.cast(((mask_preds==0) & (masks == 0)),tf.float32))
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







#gathering of the confusion matrix elemetns 
#return precision,recall,f1,iou and mae


#visualizer