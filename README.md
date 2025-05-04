# Evolution of S-boxes Using Machine Learning

This repository contains all code and setup instructions for the thesis *Evolution of S-boxes Using Machine Learning*.  
We implemented two models that predict swaps aimed at improving the cryptographic properties of S-boxes.  
These models are evaluated against randomly generated swaps, as described in detail in the thesis.

---

## Setup

- **Python version**: `3.11`
- **TensorFlow version**: `2.13.0`
- **Keras version**: `3.8.0`
- Other required packages can be found in the code (e.g., `numpy`, `matplotlib`, etc.)

---

## Dataset Generation

The script used to generate datasets is: `dataset_generate.py`  
Two datasets are already provided and were used in experiments:

- One for `4×4` S-boxes (16 entries)
- One for `8×8` S-boxes (256 entries)  
Each contains `25,040` samples.

---

## Models

We implemented two models for swap prediction:

- A **Seq2Seq** model
- A **CNN-based** model

To work with different S-box sizes (`4×4` or `8×8`), update the following parameters in the code:

- `size`: `16` for `4×4`, `256` for `8×8`
- Input layer shape: `16` for `4×4`, `256` for `8×8`
- `LEN = 2 ** x`, where `x = 4` for `4×4`, or `x = 8` for `8×8`

---

## Statistics

We provide two scripts for statistical analysis:

- `statistic_for_random.py`:  
  Computes mean accuracy and variance of random swaps improving differential spectrum.
- `t_test.py`:  
  Performs a t-test and computes p-values to compare model-generated swaps vs. random swaps.

---

## Predictor

The script `predictor.py` implements a **sequential predictor** using:

- **Random predictor** (baseline)
- **Model-based predictor**, using the best-performing trained model:  
  `my_model_9_24_52.keras`

---

## Results

- Both models showed better accuracy than random swap selection in isolated predictions.
- However, the **random sequential predictor** achieved faster convergence and better final results compared to the model-based predictor.
- We conclude that while models were able to learn meaningful patterns, limited dataset size restricted overall performance.

---

## Notes

- All scripts and assets (datasets, models, plots) are included in this directory.
- For more details and theoretical background, refer to the thesis document.

