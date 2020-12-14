from Global_parameters import *

channel_train = np.load('./data/channel_train.npy')
train_size = channel_train.shape[0]
channel_test = np.load('./data/channel_test.npy')
test_size = channel_test.shape[0]


def training_gen(bs, reshapes, SNRdb = 20):
    while True:
        index = np.random.choice(np.arange(train_size), size=bs)
        H_total = channel_train[index]
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
