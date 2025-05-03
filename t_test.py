import numpy as np
from scipy import stats


arr = [21.88, 24.44, 23.88, 23.08, 22.20, 22.20, 23.16, 23.88, 22.36, 24.04, 24.52, 23.96, 23.76, 24.36, 24.28, 24.20, 23.32, 23.88, 22.84, 23.40, 23.36, 23.96, 23.48]
x2_mean = np.mean(arr)
s2 = np.std(arr, ddof=1)
n2 = len(arr)

x1_mean = 19.96
s1 = 0.55
n1 = 1000

# Welch t-test
numerator = x1_mean - x2_mean
denominator = np.sqrt((s1**2 / n1) + (s2**2 / n2))
t_stat = numerator / denominator

# Calculation of degrees of freedom according to the Welch-Satterthwaite equation
df = ((s1**2 / n1) + (s2**2 / n2))**2 / (
    ((s1**2 / n1)**2) / (n1 - 1) + ((s2**2 / n2)**2) / (n2 - 1)
)

# p-value
p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

print(f"Mean of model results (x2̄): {x2_mean:.2f}%")
print(f"Variance of model results (s2): {s2:.2f}")
print(f"t-test: {t_stat:.4f}")
print(f"df: {df:.4f}")
print(f"p-value: {p_value:.6f}")
