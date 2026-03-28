import random
from collections import Counter
import operator
import numpy as np
from sympy import fwht
import csv
import os


# Computes the XOR difference distribution table (DDT) for the S-box `sb` with a given difference `dx`
def xprofile(sb, dx):
    N = [0] * len(sb)
    for x in range(len(sb)):
        N[sb[(x ^ dx)] ^ sb[x]] += 1
    return N

# difference distribution table (DDT)

def fullxprofile(sb):
    N = []
    for dx in range(1, len(sb)):
        N += [xprofile(sb, dx)]
    return N

# differential spectrum
def dspectrum(sb):
    p = fullxprofile(sb)
    ctr = Counter()
    for ddt in p:
        ctr += Counter([i for i in ddt[1:]])
    return sorted(ctr.items(), key=operator.itemgetter(0))

# Converts an integer `x` into a list of its binary digits
def binlist(x):
    out = [x % 2]
    while x >= 2:
        x = x // 2
        out.append(x % 2)
    return out

# Computes a binary vector `table` based on the bitwise AND of each element in `F` with `v`
def vF(v, F):
    table = [0] * len(F)
    for x in range(len(F)):
        table[x] = sum([i for i in binlist(F[x] & v)]) & 1
    return table

# Walsh-Hadamard Transform
def WHTspectrum(S):
    CM = []
    for v in range(1, len(S)):
        transform = fwht(vF(v, S))
        CM.append(transform)
    return CM

# linear spectrum
def lspectrum(sb):
    p = WHTspectrum(sb)
    ctr = Counter()
    for wht in p:
        ctr += Counter([abs(i) for i in wht[1:]])
    return sorted(ctr.items(), key=operator.itemgetter(0))

# differential uniformity
def dif_uniformity(ds):
    left_value = ds[-1][0]
    return left_value

def nonlinerity(ls):
    left_value = ls[-1][0]
    n = 2**7
    value = n - left_value
    return value

# function that find sbox properties
def get_sb_props(S):
    ds = dspectrum(S)
    #    ls = lspectrum(S)
    # du = dif_uniformity(ds)
    # nonl = nonlinerity(ls)
    da = ds[-1]

    return   ds, da

'''  Function that performs random swaps to compute the S-box and its properties. The metadata parameter 
contains experiment details. The value N=8 means it generates an 8x8 S-box (with 256 entries), 
and C specifies how many swaps will be performed during the experiment.  '''
def run_experiment_random(metadata ):
    sboxes_8 = np.load('sboxes_8.npy', allow_pickle=True)
    dataset = []
    LEN = 2 ** 8
    for i in range(len(sboxes_8)):
        S = sboxes_8[i]
        ds, da_before_swap = get_sb_props(S)
        sbox_before_swap = S.copy()
        t = 0
        while t == 0:
            S = sbox_before_swap.copy()
            x = 0
            y = 0
            while (x == y):
                x = random.randint(0, LEN - 1)
                y = random.randint(0, LEN - 1)
            # random swap
            S[x], S[y] = S[y], S[x]
            sbox_after_swap = S.copy()
            ds, da_after_swap = get_sb_props(S)
            if da_before_swap[0] > da_after_swap[0] or (
                    da_before_swap[0] == da_after_swap[0] and da_before_swap[1] > da_after_swap[1]):
                t = 1
                print("8")
                # Append data to the dataset
                dataset.append({
                    'sbox_before_swap': sbox_before_swap.tolist(),
                    'swap': (x, y),
                    'sbox_after_swap': sbox_after_swap.tolist(),
                    'da_before_swap': da_before_swap,
                    'da_after_swap': da_after_swap
                })


    for i in range(12520):
        da_ten=0
        while da_ten!=10:
            S = np.random.permutation(LEN)
            ds, da_before_swap = get_sb_props(S)
            da_ten = da_before_swap[0]

        sbox_before_swap = S.copy()
        t = 0
        while t == 0:
            S = sbox_before_swap.copy()
            x = 0
            y = 0
            while (x == y):
                x = random.randint(0, LEN - 1)
                y = random.randint(0, LEN - 1)
            # random swap
            S[x], S[y] = S[y], S[x]
            sbox_after_swap = S.copy()
            ds, da_after_swap = get_sb_props(S)
            if da_before_swap[0] > da_after_swap[0] or (
                    da_before_swap[0] == da_after_swap[0] and da_before_swap[1] > da_after_swap[1]):
                t = 1
                print(1)
                # Append data to the dataset
                dataset.append({
                    'sbox_before_swap': sbox_before_swap.tolist(),
                    'swap': (x, y),
                    'sbox_after_swap': sbox_after_swap.tolist(),
                    'da_before_swap': da_before_swap,
                    'da_after_swap': da_after_swap
                })


    # Save the dataset to a CSV file
    file_path = f"{metadata['dataset_size']}_8_dataset50_50_trueone.csv"
    file_exists = os.path.isfile(file_path)  # Check if the file already exists

    with open(file_path, mode="w", newline="") as file:  # Open in append mode
        writer = csv.DictWriter(file, fieldnames=['sbox_before_swap', 'swap', 'sbox_after_swap', 'da_before_swap',
                                                  'da_after_swap'])
        if not file_exists:  # Write header only if file doesn't exist
            writer.writeheader()
        for row in dataset:
            writer.writerow(row)


dataset_size = 25040
metadata = {'dataset_size': dataset_size}
run_experiment_random(metadata)
