#!/usr/bin/env python

from os import listdir
from os.path import isdir, join, abspath, exists
import librosa
import random
import numpy as np
from tensorflow import lite
from tensorflow.keras import layers, models
import python_speech_features
import matplotlib.pyplot as plt


# SET THE WAKE WORD HERE
wake_word = 'igor'
model_filename = 'wake_word_stop_model.h5'
feature_sets_file = 'all_targets_mfcc_sets.npz'
tflite_filename = 'wake_word_stop_lite.tflite'


# Dataset path and view possible targets
dataset_path = abspath("data/speech_commands")
if not exists(dataset_path):
    print(f"Unable to find dataset_path: {dataset_path}\n")


# Create an all targets list
print("\nFollows is a list of training words found:")
all_targets = sorted([name for name in listdir(dataset_path) if isdir(join(dataset_path, name)) and "background_noise" not in name])
if wake_word not in all_targets:
    print("Wake word not found in dataset")
    exit(187)


# See how many files are in each
num_samples = 0
for target in all_targets:
    n = len(listdir(join(dataset_path, target)))
    print(f"Samples for {target}: {n}")
    num_samples += n
    if target == wake_word:
        if n <= 0:
            print("No sounds found for wake word")
            exit(187)
        for file in listdir(join(dataset_path, target)):
            print(f"\t - {file}")
print("Total samples:", num_samples)


# Settings
target_list = all_targets
perc_keep_samples = 1.0 # 1.0 is keep all samples
val_ratio = 0.1
test_ratio = 0.1
sample_rate = 8000
num_mfcc = 16
len_mfcc = 16


# Create list of filenames along with ground truth vector (y)
filenames = []
y = []
for index, target in enumerate(target_list):
    filenames.append(listdir(join(dataset_path, target)))
    y.append(np.ones(len(filenames[index])) * index)


# Flatten filename and y vectors
filenames = [item for sublist in filenames for item in sublist]
y = [item for sublist in y for item in sublist]


# Associate filenames with true output and shuffle
filenames_y = list(zip(filenames, y))
random.shuffle(filenames_y)
filenames, y = zip(*filenames_y)
filenames = filenames[:int(len(filenames) * perc_keep_samples)]


# Calculate validation and test set sizes
val_set_size = int(len(filenames) * val_ratio)
test_set_size = int(len(filenames) * test_ratio)


# Break dataset apart into train, validation, and test sets
filenames_val = filenames[:val_set_size]
filenames_test = filenames[val_set_size:(val_set_size + test_set_size)]
filenames_train = filenames[(val_set_size + test_set_size):]


# Break y apart into train, validation, and test sets
y_orig_val = y[:val_set_size]
y_orig_test = y[val_set_size:(val_set_size + test_set_size)]
y_orig_train = y[(val_set_size + test_set_size):]


# Function: Create MFCC from given path
def calc_mfcc(path):
    # Load wavefile
    signal, fs = librosa.load(path, sr=sample_rate)

    # Create MFCCs from sound clip
    mfccs = python_speech_features.base.mfcc(signal,
        samplerate=fs,
        winlen=0.256,
        winstep=0.050,
        numcep=num_mfcc,
        nfilt=26,
        nfft=2048,
        preemph=0.0,
        ceplifter=0,
        appendEnergy=False,
        winfunc=np.hanning)
    return mfccs.transpose()


# TEST: Construct test set by computing MFCC of each WAV file
prob_cnt = 0
x_test = []
y_test = []
for index, filename in enumerate(filenames_train):

    # Stop after 500
    #if index >= 500:
    #    break

    # Create path from given filename and target item
    index_name = target_list[int(y_orig_train[index])]
    path = join(dataset_path, index_name, filename)

    # Create MFCCs
    mfccs = calc_mfcc(path)

    if mfccs.shape[1] == len_mfcc:
        x_test.append(mfccs)
        y_test.append(y_orig_train[index])
    else:
        print('Dropped:', index_name, index, mfccs.shape)
        prob_cnt += 1

print('% of problematic samples:', prob_cnt / 500)


# Function: Create MFCCs, keeping only ones of desired length
def extract_features(in_files, in_y):
    prob_cnt = 0
    out_x = []
    out_y = []

    for index, filename in enumerate(in_files):

        # Create path from given filename and target item
        index_name = target_list[int(in_y[index])]
        path = join(dataset_path, index_name, filename)

        # Check to make sure we're reading a .wav file
        if not path.endswith('.wav'):
            continue

        # Create MFCCs
        mfccs = calc_mfcc(path)

        # Only keep MFCCs with given length
        if mfccs.shape[1] == len_mfcc:
            out_x.append(mfccs)
            out_y.append(in_y[index])
        else:
            print('Dropped:', index, mfccs.shape)
            prob_cnt += 1

    return out_x, out_y, prob_cnt


# Create train, validation, and test sets
x_train, y_train, prob = extract_features(filenames_train, y_orig_train)
print('Removed percentage:', prob / len(y_orig_train))
x_val, y_val, prob = extract_features(filenames_val, y_orig_val)
print('Removed percentage:', prob / len(y_orig_val))
x_test, y_test, prob = extract_features(filenames_test, y_orig_test)
print('Removed percentage:', prob / len(y_orig_test))


# Save features and truth vector (y) sets to disk
np.savez(feature_sets_file,
         x_train=x_train,
         y_train=y_train,
         x_val=x_val,
         y_val=y_val,
         x_test=x_test,
         y_test=y_test)


# TEST: Load features
feature_sets = np.load(feature_sets_file)
print(f"feature_sets.files: {feature_sets.files}, len(feature_sets['x_train']): {len(feature_sets['x_train'])}, feature_sets['y_val']: {feature_sets['y_val']}")


# Settings
feature_sets = np.load(abspath(feature_sets_file))


# Assign feature sets
x_train = feature_sets['x_train']
y_train = feature_sets['y_train']
x_val = feature_sets['x_val']
y_val = feature_sets['y_val']
x_test = feature_sets['x_test']
y_test = feature_sets['y_test']


# Convert ground truth arrays to one wake word (1) and 'other' (0)
wake_word_index = all_targets.index(wake_word)
y_train = np.equal(y_train, wake_word_index).astype('float64')
y_val = np.equal(y_val, wake_word_index).astype('float64')
y_test = np.equal(y_test, wake_word_index).astype('float64')


# CNN for TF expects (batch, height, width, channels)
# So we reshape the input tensors with a "color" channel of 1
x_train = x_train.reshape(x_train.shape[0],
                          x_train.shape[1],
                          x_train.shape[2],
                          1)
x_val = x_val.reshape(x_val.shape[0],
                      x_val.shape[1],
                      x_val.shape[2],
                      1)
x_test = x_test.reshape(x_test.shape[0],
                        x_test.shape[1],
                        x_test.shape[2],
                        1)


# Input shape for CNN is size of MFCC of 1 sample
sample_shape = x_test.shape[1:]


# Build model
# Based on: https://www.geeksforgeeks.org/python-image-classification-using-keras/
model = models.Sequential()
model.add(layers.Conv2D(32,
                        (2, 2),
                        activation='relu',
                        input_shape=sample_shape))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Classifier
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1, activation='sigmoid'))


# Display model
model.summary()


# Add training parameters to model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['acc'])


# Train
history = model.fit(x_train,
                    y_train,
                    epochs=30,
                    batch_size=100,
                    validation_data=(x_val, y_val))


# Plot results
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()


# Save the model as a file
models.save_model(model, model_filename)


# TEST: Load model and run it against test set
model = models.load_model(model_filename)
for i in range(100, 110):
    print('Answer:', y_test[i], ' Prediction:', model.predict(np.expand_dims(x_test[i], 0)))


# Evaluate model with test set
model.evaluate(x=x_test, y=y_test)


# Convert model to TF Lite model
converter = lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
open(tflite_filename, 'wb').write(tflite_model)
print(f"Saved tflite filename to: {tflite_filename}")


print("All done")