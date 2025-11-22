#imports
import tensorflow as tf
import numpy as np
import matplotlib as plp

from data_loader import create_dataset
from sod_model import set_sod_model, optimizer, sod_loss, hard_IoU


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


def compute_metric(sod_model, test_set):
    precision_list = []
    recall_list = []
    f1_list = []
    mae_list = []
    iou_list = []






def evaluate():
    sod_model, test_set = load_model_testset()


#gathering of the confusion matrix elemetns 
#return precision,recall,f1,iou and mae


#visualizer