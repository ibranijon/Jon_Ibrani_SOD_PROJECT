from sod_model import model,sod_loss,soft_IoU,optimizer
from data_loader import create_dataset
import tensorflow as tf


def components():
    train_set, val_set, test_set = create_dataset()
    sod_model = model()
    opt = optimizer()

    return train_set,val_set,test_set,sod_model,opt

    

def train_pass():
    pass

def val_pass():
    pass

def checkpoint():
    pass

def training():
    pass