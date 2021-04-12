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

base_model_es = EarlyStopping(monitor='val_bit_err', mode='min', verbose=1, patience=10)
base_model_lr_reducer = ReduceLROnPlateau(monitor='val_bit_err', factor=np.sqrt(0.1), cooldown=0, 
                                          verbose=1, patience=10, min_lr=0.5e-6)

base_model_checkpoint = callbacks.ModelCheckpoint('./result/mlp_temp_trained_25_8_pilot_lr_no_es.h5', monitor='val_bit_err',
                                                      verbose=0, save_best_only=True, mode='min', save_weights_only=False)
checkPointList = [base_model_checkpoint]                                                  
checkPointList += [base_model_es]
checkPointList += [base_model_lr_reducer]
    
base_model = mlp(input_bits)
base_model.compile(optimizer='adam', loss='mse', metrics=[bit_err])
print("Number of layers in the base model: ",len(base_model.layers))
training_time_total = 0
training_time_start = time.time()
initial_epochs = 10
epochs = 500
base_model_info = base_model.fit(
    training_gen(1000, False, 10),
    steps_per_epoch=50,
    epochs=epochs,
    validation_data=validation_gen(1000, False, 10),
    validation_steps=1,
    callbacks = checkPointList, 
    verbose=2)
print(base_model_info.history.keys())

duration = time.time() - training_time_start
training_time_total += duration

training_average_time = training_time_total / epochs
print("average_time: ", training_average_time)
sio.savemat('./result/base_model_training_history.mat', base_model_info.history)
plot_model_history(base_model_info)

# fine tunning
model = mlp(input_bits)
model.load_weights('./result/mlp_temp_trained_25_8_pilot_lr_no_es.h5')

base_learn_rate =  base_model_info.history['lr']

adam = Adam(lr = np.array(base_learn_rate)[-1]/2)

model.trainable = True
fine_tune_at = 7
for layer in model.layers[:fine_tune_at]:
    layer.trainable =  False

model.compile(optimizer=adam, loss='mse', metrics=[bit_err])
fine_tuning_es = EarlyStopping(monitor='val_bit_err', mode='min', verbose=1, patience=20)
fine_tuning_lr_reducer = ReduceLROnPlateau(monitor='val_bit_err', factor=np.sqrt(0.1), cooldown=0, verbose=1, patience=5, min_lr=0.1e-6)
fine_tuning_checkpoint = callbacks.ModelCheckpoint('./result/fine_tuning_trained.h5', monitor='val_bit_err',
                                                    verbose=0, save_best_only=True, mode='min', save_weights_only=False)
model.summary()
print("mode trainable parameters: ", len(model.trainable_variables))

fine_tune_epochs = 100
base_epoch = base_model_info.epoch[-1]
print (base_epoch)
total_epochs =  base_epoch + fine_tune_epochs
print(total_epochs)

checkList = [fine_tuning_checkpoint]
checkList += [fine_tuning_es]
checkList += [fine_tuning_lr_reducer]

history_fine = model.fit(training_gen(1000, False, 25),
                        steps_per_epoch=50,
                        epochs=total_epochs,
                        initial_epoch = base_epoch,
                        validation_data=validation_gen(1000, False, 25),
                        validation_steps=1,
                        callbacks=checkList,
                        verbose=2)
sio.savemat('./result/fine_tuning_history.mat', history_fine.history)
plot_fine_tuning_model_history(base_model_info, history_fine)

model = mlp(input_bits)
model.load_weights('./result/fine_tuning_trained.h5')
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
print("average_time: ", test_average_time)
plt.semilogy(BER, "-*")
plt.grid(True)
plt.ylim(0.001, 0.2)
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.show()
BER_matlab = np.array(BER)
sio.savemat('./result/mlp_result_8_pilot_lr_no_es.mat', {'BER':BER_matlab, 'training_average_time':training_average_time, 'test_average_time':test_average_time})

#base_model_train(10, True, True)
#fine_tune_model(15,True, True)

