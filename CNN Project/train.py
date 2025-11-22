from sod_model import set_sod_model,sod_loss,optimizer,hard_IoU
from data_loader import create_dataset
import tensorflow as tf


def components():
    train_set, val_set, test_set = create_dataset()
    sod_model = set_sod_model()
    opt = optimizer()
    return sod_model, opt, train_set, val_set,test_set


def set_checkpoint(sod_model,opt):
    #checkpoint creater
    checkp = tf.train.Checkpoint(model=sod_model,optimizer=opt, step=tf.Variable(0))

    #manager putter of last checkpoint into the checkpoint file
    manager = tf.train.CheckpointManager(checkp, './checkpoints',max_to_keep=3)
    #if condition to check if there is anythin in the checkpoint file otherwise nuke it
    if manager.latest_checkpoint:
        print(f'Restoring from {manager.latest_checkpoint}')
        checkp.restore(manager.latest_checkpoint)
    else:
        print("There is no Checkpoint, starting from scratch")
    return checkp,manager



def train_pass(sod_model, opt, image, mask):

    #Gradient tape to record all forward pass, loss calc, gradient calc
    with tf.GradientTape() as tape:
        mask_pred = sod_model(image, training=True)#Need to understand the concept of training True and False
        loss = sod_loss(mask,mask_pred)
    
    #Calculation of gradient based on loss for each weight
    grads = tape.gradient(loss, sod_model.trainable_variables)
    #Application of gradient to its respective weight 
    opt.apply_gradients(zip(grads, sod_model.trainable_variables))

    return loss

def val_pass(sod_model, image, mask):
    #Calculation of forward pass and loss and hard IoU
    mask_pred = sod_model(image, training=False)
    loss = sod_loss(mask, mask_pred)
    hIoU = hard_IoU(mask,mask_pred)

    return loss,hIoU


def train():
    pass