import os
from collections import Counter
import operator
import random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import keras
import pandas as pd
import numpy as np
from keras import layers

# Differential Cryptanalysis Functions
def xprofile(sb, dx):
    N = [0] * len(sb)
    for x in range(len(sb)):
        N[sb[(x ^ dx)] ^ sb[x]] += 1
    return N


def fullxprofile(sb):
    N = []
    for dx in range(1, len(sb)):
        N += [xprofile(sb, dx)]
    return N


def dspectrum(sb):
    p = fullxprofile(sb)
    ctr = Counter()
    for ddt in p:
        ctr += Counter([i for i in ddt[1:]])
    return sorted(ctr.items(), key=operator.itemgetter(0))


def get_sb_props(S):
    ds = dspectrum(S)
    return ds[-1]

# Load dataset
df = pd.read_csv('25040_8_dataset.csv')

print(f"Dataset .  size: {len(df)} rows")

# Remove duplicate S-box entries
df = df.drop_duplicates(subset=['sbox_before_swap'])

print(f"Dataset cleaned. New size: {len(df)} rows")

# Extract columns
input_array = df['sbox_before_swap'].apply(lambda x: eval(x)).values  # Convert strings to lists
swap_array = df['swap'].apply(lambda x: eval(x)).values              # Convert strings to tuples
size = 256

# Define one-hot encoding helper functions
def one_hot_encode(C, rows, lines ):
    """
    One-hot encode an S-box input as a 256x256 matrix.
    Each row corresponds to the one-hot encoding of the S-box value at that position.
    """
    encoded = np.zeros((lines, rows))
    for i, val in enumerate(C):
        encoded[i, val] = 1
    return encoded


# Prepare inputs and outputs
inputs = []
outputs = []
for sbox, swap in zip(input_array, swap_array):
    inputs.append(one_hot_encode(sbox, size, size))
    outputs.append(one_hot_encode(swap, size, 2))

# Convert to numpy arrays
x = np.array(inputs)
y = np.array(outputs)

# Split into training and validation sets
split_at = len(x) - len(x) // 10
x_train, x_val = x[:split_at], x[split_at:]
y_train, y_val = y[:split_at], y[split_at:]


# Print data shapes
print("Training Data:")
print("Input shape (S-box):", x_train.shape)
print("Output shapes (Swap x, y):", y_train.shape, y_train.shape)

print("Validation Data:")
print("Input shape (S-box):", x_val.shape)
print("Output shapes (Swap x, y):", y_val.shape, y_val.shape)

print("Build model...")

# Define the number of layers and the number of units for the LSTM
num_layers = 1
input_dim = size  # One-hot encoded S-box is of size 256
output_dim = size  # One-hot encoding for the swap values is also 256

# Define the model
model = keras.Sequential()

# Add the input layer
model.add(layers.Input(shape=(256, input_dim)))  # Shape: (batch_size, MAXLEN, 256)

# First LSTM to encode the input sequence
model.add(layers.LSTM(256))  # LSTM encoding

# Repeat the output of LSTM across multiple timesteps
model.add(layers.RepeatVector(2))

# Stack multiple LSTM layers for the decoder
for _ in range(num_layers):
    model.add(layers.LSTM(256, return_sequences=True))  # Decoder LSTM

# Output layer: Two dense layers for each swap value (x and y)
model.add(layers.Dense(output_dim, activation="softmax"))  # TimeDistributed for sequential output

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=0.005), metrics=["accuracy"])

# Print the model summary to check the architecture
model.summary()

# Training parameters
epochs = 20

for epoch in range(1, epochs):
    # Train model
    model.fit(
        x_train,
        y_train,
        batch_size=64,  # Adjusted batch size
        epochs=1,
        validation_data=(x_val, y_val),

    )

    # Evaluation on validation set
    correct_count = 0  # Track correct predictions
    total_count = len(x_val)  # Total samples in validation set
    correct_count_1 = 0  # Track correct predictions
    total_count_1 = len(x_val)

    for ind in range(total_count):  # Loop over the whole validation set
        rowx, rowy = x_val[ind], y_val[ind]
        LEN = 2 ** 8
        # Convert the one-hot encoded input S-box back to its original form
        original_sbox = np.argmax(rowx, axis=1)  # Decode one-hot encoding

        preds = model.predict(np.array([rowx]), verbose=0)[0]  # Get predictions

        def get_top_two_indices(predictions):
            """Get two indexes with the highest probabilities from predictions."""
            sorted_indices = np.argsort(predictions)[::-1]  # sorted indexes based on probabilities
            return sorted_indices[0], sorted_indices[1]

        # get two best indexes
        top_x = get_top_two_indices(preds[0])
        top_y = get_top_two_indices(preds[1])

        # if they are equal choose the second index with the second-highest probability
        pred_x = top_x[0]
        pred_y = top_y[0] if top_x[0] != top_y[0] else top_y[1]

        # applying the swap
        swapped_sbox = original_sbox.copy()
        swapped_sbox[pred_x], swapped_sbox[pred_y] = swapped_sbox[pred_y], swapped_sbox[pred_x]

        df_real = get_sb_props(original_sbox)
        df_pred = get_sb_props(swapped_sbox)

        is_correct = False
        # validation if DA improved
        if df_real[0] > df_pred[0]:
            is_correct = True
        elif df_real[0] == df_pred[0] and df_real[1] > df_pred[1]:
            is_correct = True

        if is_correct:
            correct_count += 1
        # the same process for a random swap
        x = 0
        y = 0
        while x == y:
            x = random.randint(0, LEN - 1)
            y = random.randint(0, LEN - 1)
        swapped_sbox_1 = original_sbox.copy()
        # random swap
        swapped_sbox_1[x], swapped_sbox_1[y] = swapped_sbox_1[y], swapped_sbox_1[x]

        # Total samples in validation set
        df_real_1 = df_real
        df_pred_1 = get_sb_props(swapped_sbox_1)
        is_correct_1 = False

        if df_real_1[0] > df_pred_1[0]:
            is_correct_1 = True
        elif df_real_1[0] == df_pred_1[0] and df_real_1[1] > df_pred_1[1]:
            is_correct_1 = True

        if is_correct_1:
            correct_count_1 += 1

    # Compute validation accuracy
    accuracy = (correct_count / total_count) * 100

    print(f"Validation Accuracy: {accuracy:.2f}%")

    # if the accuracy is high enough save the model
    if accuracy > 24.3:
        model.save('my_model_9_24_3.keras')

    accuracy_1 = (correct_count_1 / total_count_1) * 100
    print(f"Random Accuracy: {accuracy_1:.2f}%")





