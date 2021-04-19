from Global_parameters import *

channel_train = np.load('./data/channel_train.npy')

offline_train = channel_train[0:99000]
print (offline_train)
offline_train_size = offline_train.shape[0]
online_train = channel_train[100000:]
online_train_size = online_train.shape[0]
print (online_train)

train_size = channel_train.shape[0]
channel_test = np.load('./data/channel_test.npy')
test_size = channel_test.shape[0]


def offline_training_gen(bs, reshapes, SNRdb = 20):
    while True:
        index = np.random.choice(np.arange(offline_train_size), size=bs)
        H_total = offline_train[index]
        #print ("H_total: ", H_total)
        input_samples = []
        input_labels = []
        for H in H_total:
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            signal_output, para = ofdm_simulate(bits, H, SNRdb)
            input_labels.append(bits[0:16])
            input_samples.append(signal_output)
        input_samples_array = np.asarray(input_samples)
        if (reshapes == True):
            input_samples_array_new = np.reshape(input_samples_array, (input_samples_array.shape[0], 1, input_samples_array.shape[1]))  
        else:
            input_samples_array_new = input_samples_array
        yield (input_samples_array_new, np.asarray(input_labels))

def online_training_gen(bs, reshapes, SNRdb = 20):
    while True:
        index = np.random.choice(np.arange(online_train_size), size=bs)
        H_total = online_train[index]
        #print ("H_total: ", H_total)
        input_samples = []
        input_labels = []
        for H in H_total:
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            signal_output, para = ofdm_simulate(bits, H, SNRdb)
            input_labels.append(bits[0:16])
            input_samples.append(signal_output)
        input_samples_array = np.asarray(input_samples)
        if (reshapes == True):
            input_samples_array_new = np.reshape(input_samples_array, (input_samples_array.shape[0], 1, input_samples_array.shape[1]))  
        else:
            input_samples_array_new = input_samples_array
        yield (input_samples_array_new, np.asarray(input_labels))

def validation_gen(bs, reshapes, SNRdb = 20):
    while True:
        index = np.random.choice(np.arange(test_size), size=bs)
        H_total = channel_test[index]
        input_samples = []
        input_labels = []
        for H in H_total:
            bits = np.random.binomial(n=1, p=0.5, size=(payloadBits_per_OFDM,))
            signal_output, para = ofdm_simulate(bits, H, SNRdb)
            input_labels.append(bits[0:16])
            input_samples.append(signal_output)
        input_samples_array = np.asarray(input_samples)
        if (reshapes == True):
            input_samples_array_new = np.reshape(input_samples_array, (input_samples_array.shape[0], 1, input_samples_array.shape[1]))  
        else:
            input_samples_array_new = input_samples_array
        yield (input_samples_array_new, np.asarray(input_labels))
