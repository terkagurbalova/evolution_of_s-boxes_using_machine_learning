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


# ---------- NOVÁ ČASŤ ----------
sboxes_8 = np.load('sboxes_8.npy', allow_pickle=True)

# Čítač na štatistiku
stats = Counter()

for i in range(len(sboxes_8)):
    S = sboxes_8[i]
    ds, da = get_sb_props(S)
    stats[da[0]] += 1   # počítame podľa hodnoty da[0]

# Výpis štatistiky
print("Štatistika počtu S-boxov podľa da[0]:")
for val, count in sorted(stats.items()):
    print(f"da[0] = {val}: {count} S-boxov")

print(len(sboxes_8))