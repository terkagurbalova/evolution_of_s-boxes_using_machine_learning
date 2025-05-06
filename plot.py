import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

random_accuracy = 34.65
variance = 0.87
std_dev = np.sqrt(variance)

model_accuracies = [36.26, 36.14, 35.38, 36.02, 37.10, 37.10, 35.94, 35.82, 35.58]
model_names = ["32, 0.0005","64, 0.0005","128, 0.0005", "32, 0.005","64, 0.005","128, 0.005", "32, 0.0001","64, 0.0001","128, 0.0001"]
custom_colors = ['skyblue', 'plum', '#78ab78', 'red', 'purple', 'darkblue', 'yellow', 'orange', 'brown']

# Data preparation for continuous shading
x = np.linspace(random_accuracy - 4*std_dev, 40, 500)
pdf = norm.pdf(x, random_accuracy, std_dev)

# Normalisation
pdf_normalized = pdf / max(pdf)

# Colour points for models by distance
distances = [abs(acc - random_accuracy) for acc in model_accuracies]
max_distance = max(distances)
normed_colors = [plt.cm.viridis(d / max_distance) for d in distances]

# Graph
fig, ax = plt.subplots(figsize=(10, 6))

# Gradient shading according to normal distribution
for i in range(len(x) - 1):
    ax.axhspan(x[i], x[i+1], color='gray', alpha=pdf_normalized[i] * 0.4)

# line random accuracy
ax.axhline(random_accuracy, color='black', linestyle='--', label='Random Accuracy')

# points for models
for i, (acc, name, color) in enumerate(zip(model_accuracies, model_names, custom_colors)):
    ax.scatter(i, acc, color=color, s=100, label=name)
    ax.text(i, acc + 0.3, f"{acc:.2f}", ha='center')


ax.set_xticks(range(len(model_names)))
ax.set_xticklabels(model_names)
ax.set_ylabel("Accuracy (%)")
ax.legend(loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0.)

plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.subplots_adjust(right=0.8)
plt.show()
