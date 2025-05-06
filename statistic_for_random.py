from scipy.stats import norm
import os
from collections import Counter
import operator
import random
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import tensorflow as tf
import keras

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

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

# Load dataset
df = pd.read_csv('25040_4_dataset.csv')

print(f"Dataset .  size: {len(df)} rows")

# Remove duplicate S-box entries
df = df.drop_duplicates(subset=['sbox_before_swap'])

# Save the cleaned dataset if needed
df.to_csv('cleaned_dataset.csv', index=False)

print(f"Dataset cleaned. New size: {len(df)} rows")


# Extract columns
input_array = df['sbox_before_swap'].apply(lambda x: eval(x)).values  # Convert strings to lists
swap_array = df['swap'].apply(lambda x: eval(x)).values              # Convert strings to tuples
size = 16

# Convert to numpy arrays
x = np.array(input_array)

# Split into training and validation sets
split_at = len(x) - len(x) // 10
x_train, x_val = x[:split_at], x[split_at:]

epochs =  1000
accuracy_list = []

for epoch in range(0, epochs):

    correct_count_1 = 0  # Track correct predictions
    total_count_1 = len(x_val)

    for ind in range(total_count_1):  # Loop over the whole validation set
        LEN = 2 ** 4
        # get the s-box
        original_sbox = x_val[ind]

        x = 0
        y = 0
        while x == y:
            x = random.randint(0, LEN - 1)
            y = random.randint(0, LEN - 1)
        swapped_sbox_1 = original_sbox.copy()
        # random swap
        swapped_sbox_1[x], swapped_sbox_1[y] = swapped_sbox_1[y], swapped_sbox_1[x]

        # Total samples in validation set
        df_real_1 = get_sb_props(original_sbox)
        df_pred_1 = get_sb_props(swapped_sbox_1)
        is_correct_1 = False

        if df_real_1[0] > df_pred_1[0]:
            is_correct_1 = True
        elif df_real_1[0] == df_pred_1[0] and df_real_1[1] > df_pred_1[1]:
            is_correct_1 = True


        if is_correct_1:
            correct_count_1 += 1


    accuracy_1 = (correct_count_1 / total_count_1) * 100
    accuracy_list.append(accuracy_1)
    print(f"Epoch {epoch + 1}: Random Accuracy: {accuracy_1:.2f}%")

# Calculation of mean and variance
mean_accuracy = np.mean(accuracy_list)
variance_accuracy = np.var(accuracy_list)

print(f"\nFinal Statistics:")
print(f"Mean Accuracy: {mean_accuracy:.2f}%")
print(f"Variance of Accuracy: {variance_accuracy:.2f}")

target_accuracy = 35.90

probability = norm.pdf(target_accuracy, 34.65, np.sqrt(0.87))

print(f"Probability density for accuracy {target_accuracy:.2f}%: {probability:.6f}")

