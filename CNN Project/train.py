from sod_model import set_sod_model,sod_loss,optimizer,hard_IoU
from data_loader import create_dataset
import tensorflow as tf
import os

def components():
    train_set, val_set, test_set = create_dataset()
    sod_model = set_sod_model()
    opt = optimizer()
    return sod_model, opt, train_set, val_set,test_set


def set_checkpoint(sod_model,opt):
    #Checkpoint creater
    file_path = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(file_path,'checkpoints')
    
    checkp = tf.train.Checkpoint(model=sod_model,optimizer=opt, step=tf.Variable(0))#Ruju qisaj!!!

    #Manager putter of last checkpoint into the checkpoint file
    manager = tf.train.CheckpointManager(checkp, checkpoint_path ,max_to_keep=3)
    #Condition to see if checkpoint file exists, if yes restore it
    if manager.latest_checkpoint:
        print(f'Restoring from {manager.latest_checkpoint}')
        checkp.restore(manager.latest_checkpoint)
    else:
        print("There is no Checkpoint, starting from scratch")
    return checkp,manager



def train_pass(sod_model, opt, image, mask):

    #Gradient tape to record all forward pass, loss calc, gradient calc
    with tf.GradientTape() as tape:
        mask_pred = sod_model(image, training=True)
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



nEpoch = 25

def train():
    #Set Up Model, Optimizer, Datasets, CheckPoints
    sod_model, opt, train_set, val_set, test_set = components()
    checkp, manager = set_checkpoint(sod_model,opt)
    best_val_loss = float('inf') #the values assigned its so bad that loss cant never be smaller than this, garanting your first checkpoint
    start_epoch = int(checkp.step.numpy())
    #for loop that runs n times
    for n in range(start_epoch,nEpoch):

        train_loss = []
        val_loss = []
        hIoU_list = []

        #train_pass
        for images,masks in train_set:
            loss = train_pass(sod_model,opt,images,masks)
            train_loss.append(loss)

        #val_pass
        for images,masks in val_set:
            vloss,HIoU = val_pass(sod_model,images,masks)
            val_loss.append(vloss)
            hIoU_list.append(HIoU)


        #Computer epoch averages
        avg_train_loss = sum(train_loss)/len(train_loss)
        avg_val_loss = sum(val_loss)/len(val_loss)
        avg_hIoU = sum(hIoU_list)/len(hIoU_list)


        #Logging 
        print(f'Epoch {n+1}/{nEpoch} - Train Loss:{avg_train_loss} - Val Loss:{avg_val_loss} - Val IoU:{avg_hIoU}')

        #checkpoint update
        if best_val_loss > avg_val_loss:
            best_val_loss = avg_val_loss
            checkp.step.assign(n+1)
            manager.save()
            print('Checkpoint Saved')


def main():
    train()

if __name__ == "__main__":
    main()