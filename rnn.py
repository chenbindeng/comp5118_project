from tensorflow.python.keras import *
from tensorflow.python.keras.layers import *
from generations import *
import tensorflow as tf
import matplotlib.pyplot as plt
import scipy.io as sio
import time
from utility import *


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
n_output = 16  # every 16 bit are predicted by a model
input_bits = Input(shape=(1, payloadBits_per_OFDM * 2,))

def rnn(input):
    temp = BatchNormalization()(input)
    temp = LSTM(128, dropout=0.2, recurrent_dropout=0.001, return_sequences=True)(temp)
    temp = BatchNormalization()(input)
    temp = LSTM(128, return_sequences=True)(temp)
    temp = LSTM(128)(temp)
    out_put = Dense(n_output, activation='sigmoid')(temp)
    model = Model(input_bits, out_put)
    model.compile(optimizer='adam', loss='mse', metrics=[bit_err])
    model.summary()
    return model

checkpoint = callbacks.ModelCheckpoint('./result/rnn_temp_trained_25_pilot_8.h5', monitor='val_bit_err',
                                       verbose=0, save_best_only=True, mode='min', save_weights_only=True)

model = rnn(input_bits)

training_time_total = 0
training_time_start = time.time()
epochs = 20  

model_info = model.fit_generator(
    training_gen(1000, True, 25),
    steps_per_epoch=50,
    epochs=epochs,
    validation_data=validation_gen(1000, True, 25),
    validation_steps=1,
    callbacks=[checkpoint],
    verbose=2)
print(model_info.history.keys())
duration = time.time() - training_time_start
training_time_total += duration

training_average_time = training_time_total / epochs
print("average_time: ", training_average_time)
sio.savemat('./result/rnn_model_training_history.mat', model_info.history)
plot_model_history(model_info)
model.load_weights('./result/rnn_temp_trained_25_pilot_8.h5')
test_time_total = 0
test_time_start = time.time()
BER = []
snr_in_db = np.arange(5, 30, 5)
for SNR in snr_in_db:
    y = model.evaluate(
        validation_gen(10000, True, SNR),
        steps=1
    )
    BER.append(y[1])
    print(y)
print(BER)
duration = time.time() - test_time_start
test_time_total += duration

test_average_time = test_time_total / len(snr_in_db)
print("average_time: ", test_average_time)
plt.semilogy(snr_in_db, BER, 'b-*')
plt.grid(True)
plt.ylim(0.0001, 1)
plt.xlabel('SNR (dB)')
plt.ylabel('BER')
plt.show()
BER_matlab = np.array(BER)

sio.savemat('./result/rnn_result_pilot_8.mat', {'BER':BER_matlab, 'training_average_time':training_average_time, 'test_average_time':test_average_time})
