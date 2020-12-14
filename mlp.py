from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *
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
    out_put = Dense(n_output, activation='sigmoid')(temp)
    model = Model(input_bits, out_put)
    model.compile(optimizer='adam', loss='mse', metrics=[bit_err])
    model.summary()
    return model

checkpoint = callbacks.ModelCheckpoint('./result/mlp_temp_trained_25_8_pilot.h5', monitor='val_bit_err',
                                       verbose=0, save_best_only=True, mode='min', save_weights_only=True)

model = mlp(input_bits)
training_time_total = 0
training_time_start = time.time()
epochs = 500     
model_info = model.fit_generator(
    training_gen(1000, False, 25),
    steps_per_epoch=50,
    epochs=epochs,
    validation_data=validation_gen(1000, False, 25),
    validation_steps=1,
    callbacks=[checkpoint],
    verbose=2)
print(model_info.history.keys())
duration = time.time() - training_time_start
training_time_total += duration

training_average_time = training_time_total / epochs
print("average_time: ", training_average_time)

sio.savemat('./result/mlp_model_training_history.mat', model_info.history)
plot_model_history(model_info)
model.load_weights('./result/mlp_temp_trained_25_8_pilot.h5')
test_time_total = 0
test_time_start = time.time()
BER = []
snr_in_db = np.arange(5, 30, 5)
for SNR in snr_in_db:
    y = model.evaluate(
        validation_gen(10000, False, SNR),
        steps=1
    )
    BER.append(y[1])
    print(y)
print(BER)
duration = time.time() - test_time_start
test_time_total += duration

test_average_time = test_time_total / len(snr_in_db)
print("average_time: ", test_average_time)
plt.semilogy(snr_in_db, BER)
plt.grid(True)
plt.ylim(0.0001, 1)
plt.xlabel('$E_b/N_0$ (dB)')
plt.ylabel('BER')
plt.show()
BER_matlab = np.array(BER)
sio.savemat('./result/mlp_result_8_pilot.mat', {'BER':BER_matlab, 'training_average_time':training_average_time, 'test_average_time':test_average_time})
