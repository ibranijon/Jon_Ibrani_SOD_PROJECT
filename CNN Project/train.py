from sod_model import model,sod_loss,soft_IoU,optimizer
from data_loader import create_dataset
import tensorflow as tf


def components():
    train_set, val_set, test_set = create_dataset()
    sod_model = model()
    opt = optimizer()
    return sod_model, opt, train_set, val_set,test_set


def set_checkpoint(s_model,opt):
    #checkpoint creater
    checkp = tf.train.Checkpoint(model=s_model,optimizer=opt, step=tf.Variable(0))

    #manager putter of last checkpoint into the checkpoint file
    manager = tf.train.CheckpointManager(checkp, './checkpoints',max_to_keep=3)
    #if condition to check if there is anythin in the checkpoint file otherwise nuke it
    if manager.latest_checkpoint:
        print(f'Restoring Checkpoint {manager.latest_checkpoint}')
        checkp.restore(manager.latest_checkpoint)
    else:
        print("There is no Checkpoint, starting from scratch")
    return checkp,manager


def train_pass():
    pass

def val_pass():
    pass


def train():
    pass