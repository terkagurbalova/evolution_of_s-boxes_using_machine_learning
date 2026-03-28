import os
from collections import Counter
import operator
import random
import matplotlib.pyplot as plt
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras
from keras import layers
import pandas as pd
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

def pair_to_index(i, j, size=256):
    if i > j:
        i, j = j, i
    return (2 * size - i - 1) * i // 2 + (j - i - 1)

def index_to_pair(idx, size=256):
    # Rekonštrukcia dvojice z indexu
    count = 0
    for i in range(size):
        for j in range(i + 1, size):
            if count == idx:
                return (i, j)
            count += 1
    raise ValueError("Index out of bounds")

# Load dataset
df = pd.read_csv('25040_8_dataset.csv')
sbox= df['sbox_before_swap'].apply(lambda x: eval(x)).values
sbox_df = df['da_before_swap'].apply(lambda x: eval(x)).values

model = keras.models.load_model('my_model_9_24_52.keras')

def one_hot_encode(C, rows, lines ):
    """
    One-hot encode an S-box input as a 256x256 matrix.
    Each row corresponds to the one-hot encoding of the S-box value at that position.
    """
    encoded = np.zeros((lines, rows))
    for i, val in enumerate(C):
        encoded[i, val] = 1
    return encoded

# Functions for affine transformation calculation
def ROTL8(x, shift):
    return ((x << shift) | (x >> (8 - shift))) & 0xFF

def affine_transform(q):
    return q ^ ROTL8(q, 1) ^ ROTL8(q, 2) ^ ROTL8(q, 3) ^ ROTL8(q, 4)

def random_affine_transform(sbox):
    return np.array([affine_transform(q) for q in sbox], dtype=np.uint8)

model.summary()
# Evaluation on validation set
correct_count = 0  # Track correct predictions
total_count = 25040  # Total samples in validation set
correct_count_1 = 0  # Track correct predictions
total_count_1 = 25040
correct_count = 0
correct_count_1 = 0
for ind in range(25040):
    print(ind)

#    s = sbox[ind]
#    s = np.random.permutation(256)
    s = random_affine_transform(sbox[ind])
    preds = model.predict(np.array([one_hot_encode(s, 256, 256)]), verbose=0)[0]  # Get predictions



    def get_top_two_indices(predictions):
        """Get two indexes with the highest probabilities from predictions."""
        sorted_indices = np.argsort(predictions)[::-1]  # sorted indexes based on probabilities
        return sorted_indices[0], sorted_indices[1]


    # get two best indexes
    top_x = get_top_two_indices(preds[0])
    top_y = get_top_two_indices(preds[1])

    # if they are equal choose the second index with the second-highest probability
    x = top_x[0]
    y = top_y[0] if top_x[0] != top_y[0] else top_y[1]

    # applying the swap
    swapped_sbox = s.copy()
    swapped_sbox[x], swapped_sbox[y] = swapped_sbox[y], swapped_sbox[x]

    df_real = sbox_df[ind]
    #   df_real = get_sb_props(s)
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
        x = random.randint(0, 256 - 1)
        y = random.randint(0, 256 - 1)
    swapped_sbox_1 = s.copy()
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

