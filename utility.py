from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *
from generations import *
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import os

def plot_model_history(training_channel, model_history):
    #fig, axs = plt.subplots(1,3,figsize=(8,8))
    # summarize history for accuracy
    plt.figure(figsize=(15, 8))
    plt.subplot(1,3,1)
    plt.semilogy(range(1, len(model_history.history['bit_err']) + 1), model_history.history['bit_err'])
    plt.semilogy(range(1, len(model_history.history['val_bit_err']) + 1), model_history.history['val_bit_err'])
    plt.ylabel('Bit Error')
    #plt.xticks(np.arange(1,len(model_history.history['bit_err'])+1),len(model_history.history['bit_err'])/10)
    plt.legend(['Training', 'validation'], loc='best')
    plt.grid(True, which='both')
    plt.xlabel('Epoch')

    # summarize history for loss
    plt.subplot(1,3,2)
    plt.semilogy(range(1, len(model_history.history['loss']) + 1), model_history.history['loss'], label = 'Training')
    plt.semilogy(range(1, len(model_history.history['val_loss']) + 1), model_history.history['val_loss'], label = 'Validation')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.grid(True, which="both")
    plt.xlabel('Epoch')

    plt.subplot(1,3,3)
    plt.semilogy(range(1,len(model_history.history['lr'])+1), model_history.history['lr'])
    print("bit_err min: ", np.min(model_history.history['bit_err']))
    print("val_bit_err: ", np.min(model_history.history['val_bit_err']))
    print("loss: ", np.min(model_history.history['loss']))
    print("val_loss: ", np.min(model_history.history['val_loss']))
    print("lr min: ", np.min(model_history.history['lr']))
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.legend(['Learning Rate'], loc='best')
    plt.grid(True, which="both")

    plt.savefig('./plot/base_model_training_' + str(training_channel) + 'dB.jpg')


def plot_model_history_with_no_lr(model_history):
    #fig, axs = plt.subplots(1,3,figsize=(8,8))
    # summarize history for accuracy
    plt.figure(figsize=(15, 8))
    plt.subplot(2,1,1)
    plt.semilogy(range(1,len(model_history.history['bit_err'])+1),model_history.history['bit_err'])
    plt.semilogy(range(1,len(model_history.history['val_bit_err'])+1),model_history.history['val_bit_err'])
    plt.title('Model Bit Error')
    plt.ylabel('Bit Error')
    plt.xlabel('Epochs')
    plt.xticks(np.arange(1,len(model_history.history['bit_err'])+1),len(model_history.history['bit_err'])/10)
    plt.legend(['training', 'validation'], loc='best')
    plt.grid(True, which="major")

    # summarize history for loss
    plt.subplot(2,1,2)
    plt.semilogy(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    plt.semilogy(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    plt.ylabel('Loss')
    plt.xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    plt.legend(['training', 'validation'], loc='best')
    plt.grid(True, which="major")

    plt.show()


def plot_fine_tuning_model_history(base_model_channel, transfer_channel, base_model_info, history_fine):
    bit_err = base_model_info['bit_err']
    print ("bit_err: ", bit_err)
    val_bit_err = base_model_info['val_bit_err']

    loss = base_model_info['loss']
    val_loss = base_model_info['val_loss']

    lr = base_model_info['lr']

    base_model_epoch = len(lr)

    plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1)
    plt.semilogy(range(1, len(bit_err) + 1), bit_err, label='Training')
    plt.semilogy(range(1, len(val_bit_err) + 1), val_bit_err, label='Validation')
    plt.legend(loc='best')
    plt.ylabel('bit error')
    #plt.ylim([min(plt.ylim()),0.3])
    plt.grid(True, which='both')
    #plt.title('Training and Validation bit error')

    plt.subplot(3, 1, 2)
    plt.semilogy(range(1, len(loss) + 1), loss, label='Training Loss')
    plt.semilogy(range(1, len(val_loss) + 1), val_loss, label='Validation Loss')
    plt.legend(loc='best')
    plt.ylabel('loss')
    plt.grid(True, which='both')
    #plt.title('Training and Validation Loss')

    plt.subplot(3,1,3)
    plt.semilogy(range(1, len(lr) + 1), lr)
    plt.xlabel('Epoch')
    plt.xlabel('Learning Rate')
    plt.legend(loc='best')
    plt.grid(True, which="both")

    plt.savefig('./plot/base_model_' + str(base_model_channel) + '_dB.jpg')
    
    bit_err += history_fine.history['bit_err']

    val_bit_err += history_fine.history['val_bit_err']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    lr +=  history_fine.history['lr']

    plt.figure(figsize=(15, 15))
    plt.subplot(3, 1, 1)
    plt.semilogy(range(1, len(bit_err) + 1), bit_err, label='Training')
    plt.semilogy(range(1, len(val_bit_err) + 1), val_bit_err, label='Validation')
    plt.plot([base_model_epoch, base_model_epoch],
             plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='best')
    plt.grid(True, which='both')
    plt.ylabel('Bit Error')
    plt.xlabel('epoch')
    plt.title('Training and bit error')

    plt.subplot(3, 1, 2)
    plt.semilogy(range(1, len(loss) + 1), loss, label='Training')
    plt.semilogy(range(1, len(val_loss) + 1), val_loss, label='Validation')
    #plt.ylim([0, 1.0])
    plt.plot([base_model_epoch, base_model_epoch],
             plt.ylim(),label='Start Fine Tuning')
    plt.legend(loc='best')
    #plt.title('Training and Validation Loss')
    #plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.grid(True, which='both')

    plt.subplot(3, 1, 3)
    plt.semilogy(range(1, len(lr) + 1), lr, label='learning rate')
    #plt.ylim([0, 1.0])
    plt.plot([base_model_epoch, base_model_epoch],
             plt.ylim(), 'g-', label='Start Fine Tuning')
    plt.legend(loc='best')
    #plt.title('ALR')
    plt.xlabel('epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, which='both')
    plt.savefig('./plot/fine_tune_model_training_history_base_' + str(base_model_channel) + '_TL_' + str(transfer_channel) + '_dB.jpg')

    print("Base model peroramce: ")
    print("base model bit_err min: ", np.min(base_model_info['bit_err']))
    print("base model val_bit_err: ", np.min(base_model_info['val_bit_err']))
    print("base model loss: ", np.min(base_model_info['loss']))
    print("base model val_loss: ", np.min(base_model_info['val_loss']))
    print("base model lr: ", np.min(base_model_info['lr']))

    print("\n\nTransfer Learning Performance: ")
    print("fine tuning bit_err min: ", np.min(history_fine.history['bit_err']))
    print("fine tuning val_bit_err: ", np.min(history_fine.history['val_bit_err']))
    print("fine tuning loss: ", np.min(history_fine.history['loss']))
    print("fine tuning val_loss: ", np.min(history_fine.history['val_loss']))
    print("fine tuning lr: ", np.min(history_fine.history['lr']))