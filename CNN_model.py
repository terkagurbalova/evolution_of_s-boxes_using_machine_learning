import os
from collections import Counter
import operator
import random
import keras
import pandas as pd
import numpy as np
from keras import layers, Model, backend

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
df = pd.read_csv('25040_4_dataset.csv')

# Extract columns
input_array = df['sbox_before_swap'].apply(lambda x: eval(x)).values  # Convert strings to lists
swap_array = df['swap'].apply(lambda x: eval(x)).values              # Convert strings to tuples

size = 16

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

x = np.array(inputs)
y = np.array(outputs)

# Split into training and validation sets
split_at = len(x) - len(x) // 10
x_train, x_val = x[:split_at], x[split_at:]
y_train, y_val = y[:split_at], y[split_at:]

for kl in range(3):

    # Model
    input_layer = layers.Input(shape=(size, size, 1))

    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
    x = layers.MaxPooling2D((2, 2))(x)

    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)

    # Outputs
    swap_1 = layers.Dense(size, activation='softmax', name='swap_1')(x)
    concat_input = layers.Concatenate()([x, swap_1])
    dense_intermediate = layers.Dense(size, activation='relu')(concat_input)
    swap_2 = layers.Dense(size, activation='softmax', name='swap_2')(dense_intermediate)

    model = Model(inputs=input_layer, outputs=[swap_1, swap_2])

    model.compile(
        loss='categorical_crossentropy',
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        metrics=['accuracy', 'accuracy']
    )

    model.summary()


    # Training parameters
    epochs = 20
    batch_size = [32, 64, 128]

    for epoch in range(1, epochs):
        # Train the model
        model.fit(
            x_train,
            [y_train[:, 0], y_train[:, 1]],
            validation_data=(x_val, [y_val[:, 0], y_val[:, 1]]),
            epochs=1,
            batch_size=batch_size[kl],
        )

        # Evaluate model
        val_results = model.evaluate(x_val, [y_val[:, 0], y_val[:, 1]])
        val_loss = val_results[0]
        swap_1_loss = val_results[1]
        swap_2_loss = val_results[2]
        swap_1_acc = val_results[3]
        swap_2_acc = val_results[4]
        print(f"Validation Accuracy - swap_1: {swap_1_acc:.2%}, swap_2: {swap_2_acc:.2%}")

        avg_acc = (swap_1_acc + swap_2_acc) / 2
        print(f"Average Validation Accuracy: {avg_acc:.2%}")

        swap1_class_counts = np.sum(y_val[:, 0], axis=0)
        swap2_class_counts = np.sum(y_val[:, 1], axis=0)

        print("swap_1 distribúcia:", np.argmax(swap1_class_counts), np.max(swap1_class_counts))
        print("swap_2 distribúcia:", np.argmax(swap2_class_counts), np.max(swap2_class_counts))

        correct_count = 0  # Track correct predictions
        total_count = len(x_val)  # Total samples in validation set
        correct_count_1 = 0  # Track correct predictions
        total_count_1 = len(x_val)

        for ind in range(total_count):  # Loop over the whole validation set
            rowx, rowy = x_val[ind], y_val[ind]
            LEN = 2 ** 4
            # Convert the one-hot encoded input S-box back to its original form
            original_sbox = np.argmax(rowx, axis=1)  # Decode one-hot encoding
            preds_1, preds_2 = model.predict(np.array([rowx]), verbose=0)

            # Most likely indexes
            top_x = np.argsort(preds_1[0])[::-1]
            top_y = np.argsort(preds_2[0])[::-1]

            # If they are equal, choose the second best value
            pred_x = top_x[0]
            pred_y = top_y[0] if top_x[0] != top_y[0] else top_y[1]

            #The S-box before and after applying the predicted swap
            swapped_sbox = original_sbox.copy()
            swapped_sbox[pred_x], swapped_sbox[pred_y] = swapped_sbox[pred_y], swapped_sbox[pred_x]

            df_real = get_sb_props(original_sbox)
            df_pred = get_sb_props(swapped_sbox)

            is_correct = False

            if df_real[0] > df_pred[0]:
                is_correct = True
            elif df_real[0] == df_pred[0] and df_real[1] > df_pred[1]:
                is_correct = True


            if is_correct:
                correct_count += 1

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

        accuracy_1 = (correct_count_1 / total_count_1) * 100
        print(f"Random Accuracy: {accuracy_1:.2f}%")
