from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from generations import *
import tensorflow as tf
from utility import *
import matplotlib.pyplot as plt
import scipy.io as sio
import time
import os

base_model_train_channel = 10
result = open('./result/running_result'+str(base_model_train_channel)+'.log', 'w')

def bit_err(y_true, y_pred):
    err = 1 - tf.reduce_mean(
        tf.reduce_mean(
            tf.cast(
                tf.equal(
                    tf.sign(
                        y_pred - 0.5),
                    tf.cast(
                        tf.sign(
                            y_true - 0.5),
                        tf.float32)), dtype=tf.float32),
            1))
    return err

n_hidden_1 = 500
n_hidden_2 = 250  # 1st layer num features
n_hidden_3 = 120  # 2nd layer num features
n_output = 16  # every 16 bit are predicted by a model

input_bits = Input(shape=(payloadBits_per_OFDM * 2,))

def mlp(input):
    temp = BatchNormalization()(input)
    temp = Dense(n_hidden_1, activation='relu')(temp)
    temp = BatchNormalization()(temp)
    temp = Dense(n_hidden_2, activation='relu')(temp)
    temp = BatchNormalization()(temp)
    temp = Dense(n_hidden_3, activation='relu')(temp)
    temp = BatchNormalization()(temp)
    temp = Dense(n_output, activation='relu')(temp)
    temp = BatchNormalization()(temp)
    out_put = Dense(n_output, activation='sigmoid')(temp)
    model = Model(input_bits, out_put)
    return model

def base_model_train(base_channel, early_stop, adaptive_LR):
    base_model_es = EarlyStopping(monitor='val_bit_err', mode='min', verbose=1, patience=10)
    base_model_lr_reducer = ReduceLROnPlateau(monitor='val_bit_err', factor=np.sqrt(0.1), cooldown=0, 
                                              verbose=1, patience=20, min_lr=0.5e-6)

    base_model_checkpoint = callbacks.ModelCheckpoint('./result/mlp_base_model_'+str(base_channel)+'_bs1000.h5', monitor='val_bit_err',
                                                      verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    checkPointList = [base_model_checkpoint]                                                  
    if (early_stop==True):
        checkPointList += [base_model_es]
    if (adaptive_LR == True):
        checkPointList += [base_model_lr_reducer]
    
    base_model = mlp(input_bits)
    base_model.compile(optimizer='adam', loss='mse', metrics=[bit_err])
    training_time_total = 0
    training_time_start = time.time()
    epochs = 500
    base_model_info = base_model.fit(
        offline_training_gen(1000, False, base_channel),
        steps_per_epoch=50,
        epochs=epochs,
        validation_data=validation_gen(1000, False, base_channel),
        validation_steps=1,
        callbacks = checkPointList, 
        verbose=2)

    duration = time.time() - training_time_start
    training_time_total += duration

    training_average_time = training_time_total / epochs

    np.save('./result/base_model_training_snr_'+str(base_channel)+'_1000_history.npy', base_model_info.history)
    print('epoch: ', base_model_info.epoch)
    np.save('./result/base_model_'+str(base_channel)+'_epoch.npy', base_model_info.epoch)
    sio.savemat('./result/base_model_epoch.mat', {'epoch':base_model_info.epoch[-1]})

    result.write("Base model training summary: \n")
    result.write("bit_err min: " + str(np.min(base_model_info.history['bit_err'])) + "\n")
    result.write("val_bit_err: " + str(np.min(base_model_info.history['val_bit_err'])) + "\n")
    result.write("loss: " + str(np.min(base_model_info.history['loss'])) + "\n")
    result.write("lr min: " + str(np.min(base_model_info.history['lr'])) + "\n" )
    result.write("Average Training Time: " + str(training_average_time) + "\n")

    if (adaptive_LR == True):
        plot_model_history(base_channel, base_model_info.history)
    else:
        plot_model_history_with_no_lr(base_model_info)


def fine_tune_model(base_channel, real_channel, early_stop, adaptive_learning_rate):
    # fine tunning
    model = mlp(input_bits)
    model.load_weights('./result/mlp_base_model_'+str(base_channel)+'_bs1000.h5')

    base_model_info = np.load('./result/base_model_training_snr_'+str(base_channel)+'_1000_history.npy',allow_pickle='TRUE').item()

    learn_rate =  base_model_info['lr']
    print("lr:", learn_rate)

    base_model_epoch = np.load('./result/base_model_'+str(base_channel)+'_epoch.npy',allow_pickle='TRUE')

    print('base_model_epoch: ', base_model_epoch)

    adam = Adam(lr = learn_rate[-1]/2)
    model.compile(optimizer=adam, loss='mse', metrics=[bit_err])

    model.trainable = True
    fine_tune_at = 7
    for layer in model.layers[:fine_tune_at]:
        layer.trainable =  False

    model.compile(optimizer=adam, loss='mse', metrics=[bit_err])
    fine_tuning_es = EarlyStopping(monitor='val_bit_err', mode='min', verbose=1, patience=15)
    fine_tuning_lr_reducer = ReduceLROnPlateau(monitor='val_bit_err', factor=np.sqrt(0.1), cooldown=0, verbose=1, patience=5, min_lr=0.1e-6)
    fine_tuning_checkpoint = callbacks.ModelCheckpoint('./result/fine_tuning_trained_'+ str(real_channel)+'.h5', monitor='val_bit_err',
                                                        verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    model.summary()
    print("mode trainable parameters: ", len(model.trainable_variables))

    fine_tune_epochs = 100
    base_epoch = base_model_epoch[-1]
    total_epochs =  base_epoch + fine_tune_epochs
    print(total_epochs)

    checkList = [fine_tuning_checkpoint]

    if (early_stop == True):
        checkList += [fine_tuning_es]
    if (adaptive_learning_rate == True):
        checkList += [fine_tuning_lr_reducer]
    fine_tune_time_total = 0
    fine_tune_start = time.time()

    history_fine = model.fit(online_training_gen(1000, False, real_channel),
                            steps_per_epoch=50,
                            epochs=total_epochs,
                            initial_epoch = base_epoch,
                            validation_data=validation_gen(1000, False, real_channel),
                            validation_steps=1,
                            callbacks=checkList,
                            verbose=2)

    np.save('./result/fine_tuning_'+ str(real_channel) +'_history.npy',  history_fine.history)
    duration = time.time() - fine_tune_start
    fine_tune_time_total += duration
    fine_tune_average_time = fine_tune_time_total / (history_fine.epoch[-1] - base_epoch)
    print('fine_tune_average time: ', fine_tune_average_time)
    plot_fine_tuning_model_history(base_channel, real_channel, base_model_info, history_fine)

    result.write("Transfer Learning Performance for SNR: " + str(real_channel)+ " dB\n")
    result.write("fine tuning bit_err min: " + str(np.min(history_fine.history['bit_err'])) + "\n")
    result.write("fine tuning val_bit_err: " + str(np.min(history_fine.history['val_bit_err'])) + "\n")
    result.write("fine tuning loss: " + str(np.min(history_fine.history['loss'])) + "\n")
    result.write("fine tuning val_loss: " + str(np.min(history_fine.history['val_loss'])) + "\n")
    result.write("fine tuning lr: " + str(np.min(history_fine.history['lr'])) + "\n")
    result.write('fine_tune_average time: ' + str(fine_tune_average_time) + '\n')


def prediction(base_channel, channel, transfer_learning):
    model = mlp(input_bits)
    if (transfer_learning == True):
        model.load_weights('./result/fine_tuning_trained_' + str(channel) +'.h5')
    else:
        model.load_weights('./result/mlp_base_model_' + str(base_channel) +'.h5')
    model.compile(optimizer='adam', loss='mse', metrics=[bit_err])
    test_time_total = 0
    test_time_start = time.time()
    BER = []
    snr_in_db = np.arange(5, 30, 5)
    if (transfer_learning == True):
        for i in snr_in_db:
            y = model.evaluate(
                validation_gen(10000, False, channel),
                steps=1
                )
            BER.append(y[1])
            print(y)
    else:
        for snr in snr_in_db:
            y = model.evaluate(
                validation_gen(10000, False, snr),
                steps=1
                )
            BER.append(y[1])
            print(y)
    print(BER)
    duration = time.time() - test_time_start
    test_time_total += duration

    test_average_time = test_time_total / len(snr_in_db)
    result.write("average_time: " + str(test_average_time) + "\n")
    result.write("online decode performance: base channl " + str(base_channel) + " dB + TR channe " +str(channel) + " dB\n")
    result.write("BER: " + str(BER) + "\n\n")
    plt.figure(figsize=(15, 8))
    plt.semilogy(BER, "-*")
    plt.grid(True)
    plt.ylim(0.001, 0.2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.savefig('./plot/decoding_with_transfer_learning_base_' + str(base_channel) + '_tr_'+str(channel)+'.jpg')
    #plt.show()
    BER_matlab = np.array(BER)
    np.save('./result/prediction_result_' + str(channel) +'_dB.npy', BER)
    #sio.savemat('./result/mlp_result_8_pilot_lr_no_es.mat', {'BER':BER_matlab, 'training_average_time':training_average_time, 'test_average_time':test_average_time})


def online_decoding(base_channel):
    model = mlp(input_bits)
    model.load_weights('./result/mlp_base_model_'+str(base_channel)+'1.h5')
    model.compile(optimizer='adam', loss='mse', metrics=[bit_err])
    test_time_total = 0
    test_time_start = time.time()
    BER = []
    snr_in_db = np.arange(5, 30, 5)
    for snr in snr_in_db:
        y = model.evaluate(
            validation_gen(10000, False, snr),
            steps=1
        )
        BER.append(y[1])
        print(y)
    print(BER)
    duration = time.time() - test_time_start
    test_time_total += duration

    test_average_time = test_time_total / len(snr_in_db)
    result.write("average_time: " + str(test_average_time) + "\n")
    result.write("online decode performance: base channl " + str(base_channel) + " dB\n")
    result.write("BER: " + str(BER) + "\n\n")
    plt.figure(figsize=(15, 8))
    plt.semilogy(BER, "-*")
    plt.grid(True)
    plt.ylim(0.001, 0.2)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.savefig('./plot/online_decode.jpg')
    BER_matlab = np.array(BER)
    np.save('./result/prediction_result_' + str(base_channel) +'_dB.npy', BER)
    #sio.savemat('./result/mlp_result_8_pilot_lr_no_es.mat', {'BER':BER_matlab, 'training_average_time':training_average_time, 'test_average_time':test_average_time})

#base_model_train(base_model_train_channel, False, True)   #model training with no early stop
base_model_train(base_model_train_channel, True, True)
snr_in_db = np.arange(5, 30, 5)
for transfer_channel in snr_in_db:
    fine_tune_model(base_model_train_channel,transfer_channel,True, True)

    prediction(base_model_train_channel, transfer_channel, True)
#base_model_info = np.load('./result/base_model_training_snr_'+str(base_model_train_channel)+'_history.npy',allow_pickle='TRUE').item()
#plot_base_model_history(base_model_train_channel, base_model_info)
#online_decoding(base_model_train_channel)

result.close()