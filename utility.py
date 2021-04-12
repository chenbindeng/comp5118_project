from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *
from generations import *
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import os

def plot_model_history(model_history):
    fig, axs = plt.subplots(1,3,figsize=(15,5))
    # summarize history for accuracy
    axs[0].semilogy(range(1,len(model_history.history['bit_err'])+1),model_history.history['bit_err'])
    axs[0].semilogy(range(1,len(model_history.history['val_bit_err'])+1),model_history.history['val_bit_err'])
    axs[0].set_title('Model Bit Error')
    axs[0].set_ylabel('Bit Error')
    axs[0].set_xlabel('Epochs')
    axs[0].set_xticks(np.arange(1,len(model_history.history['bit_err'])+1),len(model_history.history['bit_err'])/10)
    axs[0].legend(['training', 'validation'], loc='best')
    axs[0].grid(True, which="both")
    # summarize history for loss
    axs[1].semilogy(range(1,len(model_history.history['loss'])+1),model_history.history['loss'])
    axs[1].semilogy(range(1,len(model_history.history['val_loss'])+1),model_history.history['val_loss'])
    axs[1].set_title('Model Loss')
    axs[1].set_ylabel('Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_xticks(np.arange(1,len(model_history.history['loss'])+1),len(model_history.history['loss'])/10)
    axs[1].legend(['training', 'validation'], loc='best')
    axs[1].grid(True, which="both")
    axs[2].set_title('Learning')
    axs[2].semilogy(range(1,len(model_history.history['lr'])+1),model_history.history['lr'])
    print("bit_err min: ", np.min(model_history.history['bit_err']))
    print("val_bit_err: ", np.min(model_history.history['val_bit_err']))
    print("loss: ", np.min(model_history.history['loss']))
    print("val_loss: ", np.min(model_history.history['val_loss']))
    print("lr min: ", np.min(model_history.history['lr']))
    axs[2].set_ylabel('Learning Rate')
    axs[2].set_xlabel('Epochs')
    axs[2].set_xticks(np.arange(1,len(model_history.history['lr'])+1),len(model_history.history['lr'])/10)
    axs[2].legend(['Learning Rate'], loc='best')
    axs[2].grid(True, which="both")

    plt.show()

def plot_fine_tuning_model_history(base_model_info, history_fine):
    bit_err = base_model_info.history['bit_err']
    val_bit_err = base_model_info.history['val_bit_err']

    loss = base_model_info.history['loss']
    val_loss = base_model_info.history['val_loss']

    lr = base_model_info.history['lr']

    base_model_epoch = len(lr)

    plt.figure(figsize=(8, 8))
    plt.subplot(3, 1, 1)
    plt.semilogy(bit_err, label='Training bit error')
    plt.semilogy(val_bit_err, label='Validation bit error')
    plt.legend(loc='upper right')
    plt.ylabel('bit error')
    plt.ylim([min(plt.ylim()),0.2])
    plt.title('Training and Validation bit error')

    plt.subplot(3, 1, 2)
    plt.semilogy(loss, label='Training Loss')
    plt.semilogy(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    plt.title('Training and Validation Loss')
    plt.xlabel('epoch')
    plt.show()
    
    bit_err += history_fine.history['bit_err']

    val_bit_err += history_fine.history['val_bit_err']

    loss += history_fine.history['loss']
    val_loss += history_fine.history['val_loss']

    lr +=  history_fine.history['lr']

    plt.figure(figsize=(8, 8))
    plt.subplot(3, 1, 1)
    plt.semilogy(bit_err, label='Training bit error')
    plt.semilogy(val_bit_err, label='Validation bit error')
    #plt.ylim([0.8, 1])
    plt.plot([base_model_epoch-1, base_model_epoch-1],
          plt.ylim(), label='Start Fine Tuning')
    plt.legend(loc='upper right')
    plt.grid(True, which='major')
    plt.ylabel('Bit Error')
    #plt.xlabel('epoch')
    plt.title('Training and bit error')

    plt.subplot(3, 1, 2)
    plt.semilogy(loss, label='Training Loss')
    plt.semilogy(val_loss, label='Validation Loss')
    #plt.ylim([0, 1.0])
    plt.plot([base_model_epoch-1, base_model_epoch-1],
             plt.ylim(),label='Start Fine Tuning')
    plt.legend(loc='upper right')
    #plt.title('Training and Validation Loss')
    #plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True, which='major')

    plt.subplot(3, 1, 3)
    plt.semilogy(lr, label='Adaptive Learning Rate')
    #plt.ylim([0, 1.0])
    plt.plot([base_model_epoch-1, base_model_epoch-1],
             plt.ylim(),label='Start Fine Tuning')
    plt.legend(loc='upper right')
    #plt.title('ALR')
    plt.xlabel('epoch')
    plt.ylabel('Learning Rate')
    plt.grid(True, which='major')
    plt.show()