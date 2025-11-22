#imports
import tensorflow as tf
import numpy as np
import matplotlib as plp

from data_loader import create_dataset
from sod_model import set_sod_model, optimizer, sod_loss, hard_IoU

#for loop executing the train function on the test set

#gathering of the confusion matrix elemetns 
#return precision,recall,f1,iou and mae


#visualizer