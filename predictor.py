import os
from collections import Counter
import operator
import random
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)
import numpy as np

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

# Functions for affine transformation calculation
def ROTL8(x, shift):
    return ((x << shift) | (x >> (8 - shift))) & 0xFF

def affine_transform(q):
    return q ^ ROTL8(q, 1) ^ ROTL8(q, 2) ^ ROTL8(q, 3) ^ ROTL8(q, 4)

def random_affine_transform(sbox):
    return np.array([affine_transform(q) for q in sbox], dtype=np.uint8)

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


model = keras.models.load_model('my_model_9_24_52.keras')

LEN = 2 ** 8
sboxes_model = []
da_random_for_10 = [[] for _ in range(100)]
da_random_for_8 = [[] for _ in range(100)]
da_model_for_10 = [[] for _ in range(100)]
da_model_for_8 = [[] for _ in range(100)]
for round in range(100):
    # generating 100 8x8 s-boxes with differential spectrum value 10
    hehe = 0 #  indicates if differential spectrum value is 10
    while hehe==0:
        # get 8x8 sbox
        S = np.random.permutation(LEN)
        DA = get_sb_props(S)
        if DA[0]==10:
            sboxes_model.append(S)
            hehe=1

sboxes_random = sboxes_model.copy()
'''
Random predictor that generate random x and y values used as swap 1000times for each s-box
'''
for s in range(100):
    r = 0
    while r<1000:
        x = random.randint(0, LEN - 1)
        y = random.randint(0, LEN - 1)

        if x == y:
            continue
        r = r + 1

        df_real = get_sb_props(sboxes_random[s])
        swapped_sbox = sboxes_random[s].copy()
        swapped_sbox[x], swapped_sbox[y] = swapped_sbox[y], swapped_sbox[x]
        df_pred = get_sb_props(swapped_sbox)

        is_correct = False

        if df_real[0] > df_pred[0]:
            is_correct = True
        elif df_real[0] == df_pred[0] and df_real[1] > df_pred[1]:
            is_correct = True
        if is_correct:
            sboxes_random[s] = swapped_sbox
            if df_pred[0]==10:
                da_random_for_10[s].append(df_pred[1])
                da_random_for_8[s].append(128)
            if df_pred[0] == 8:
                da_random_for_10[s].append(0)
                da_random_for_8[s].append(df_pred[1])

        else:
            if df_real[0]==10:
                da_random_for_10[s].append(df_real[1])
                da_random_for_8[s].append(128)
            if df_real[0] == 8:
                da_random_for_10[s].append(0)
                da_random_for_8[s].append(df_real[1])


# Now calculate mean per swap
mean_per_swap_10 = np.nanmean(da_random_for_10, axis=0)
mean_per_swap_8 = np.nanmean(da_random_for_8, axis=0)


'''
Model predictor that generate x and y values used as swap 1000times for each s-box
'''

for s in range(100):
    m = 0
    preds = model.predict(np.array([one_hot_encode(sboxes_model[s], size, size)]), verbose=0)[0]  # Get predictions

    # Predictions of model
    swap1_probs = preds[0]  # size (256,)
    swap2_probs = preds[1]  # size (256,)

    # Create a matrix of probabilities
    total_probs = swap1_probs[:, None] * swap2_probs[None, :]  # (256, 256) matrix

    # Set the probability for i == j to a very small value, so that they are never selected
    np.fill_diagonal(total_probs, -1)

    # 1D array
    flat_probs = total_probs.flatten()

    # get 1000 highest probabilities
    top_indices = np.argsort(flat_probs)[-1000:][::-1]

    # Conversion back to (i, j) pairs
    top_swaps = [divmod(index, 256) for index in top_indices]

    while m < 1000:

        x = top_swaps[m][0]
        y = top_swaps[m][1]

        if x == y:
            continue
        m = m + 1

        df_real = get_sb_props(sboxes_model[s])
        swapped_sbox = sboxes_model[s].copy()
        swapped_sbox[x], swapped_sbox[y] = swapped_sbox[y], swapped_sbox[x]
        df_pred = get_sb_props(swapped_sbox)

        is_correct = False

        if df_real[0] > df_pred[0]:
            is_correct = True
        elif df_real[0] == df_pred[0] and df_real[1] > df_pred[1]:
            is_correct = True
        if is_correct:
            sboxes_model[s] = random_affine_transform(swapped_sbox)
            # m = 0
            preds = model.predict(np.array([one_hot_encode(sboxes_model[s], size, size)]), verbose=0)[
                0]  # Get predictions

            swap1_probs = preds[0]
            swap2_probs = preds[1]
            total_probs = swap1_probs[:, None] * swap2_probs[None, :]
            np.fill_diagonal(total_probs, -1)
            flat_probs = total_probs.flatten()
            top_indices = np.argsort(flat_probs)[-1000:][::-1]
            top_swaps = [divmod(index, 256) for index in top_indices]
            if df_pred[0] == 10:
                da_model_for_10[s].append(df_pred[1])
                da_model_for_8[s].append(128)
            if df_pred[0] == 8:
                da_model_for_10[s].append(0)
                da_model_for_8[s].append(df_pred[1])

        else:
            if df_real[0] == 10:
                da_model_for_10[s].append(df_real[1])
                da_model_for_8[s].append(128)
            if df_real[0] == 8:
                da_model_for_10[s].append(0)
                da_model_for_8[s].append(df_real[1])


# Now calculate mean per swap
mean_modelper_swap_10 = np.nanmean(da_model_for_10, axis=0)
mean_modelper_swap_8 = np.nanmean(da_model_for_8, axis=0)

print(mean_modelper_swap_10)
print(mean_modelper_swap_8)

# --- Plotting Section ---

# Figure 1: DA = 10
plt.figure(figsize=(10, 5))
plt.plot(mean_modelper_swap_10, label='DA Model for 10', color='blue', marker='o')
plt.plot(mean_per_swap_10, label='DA Random for 10', color='green', marker='o')
plt.xlabel('Swap index')
plt.ylabel('Average value across S-boxes')
plt.title('Average DA Model and Random Values Over Swaps (DA = 10)')
plt.legend()
plt.grid(True)
plt.show()

# Figure 2: DA = 8
plt.figure(figsize=(10, 5))
plt.plot(mean_modelper_swap_8, label='DA Model for 8', color='blue', marker='x')
plt.plot(mean_per_swap_8, label='DA Random for 8', color='green', marker='x')
plt.xlabel('Swap index')
plt.ylabel('Average value across S-boxes')
plt.title('Average DA Model Values Over Swaps (DA = 8)')
plt.legend()
plt.grid(True)
plt.show()