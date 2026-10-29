import numpy as np

np.random.seed(42)

X = np.random.exponential(scale=1.0, size=10000)
n = len(X)
B = 2000


def mad(x):
    return np.median(np.abs(x - np.median(x)))


def variance(x):
    return np.var(x, ddof=1)


mad_boot = []
var_boot = []
for _ in range(B):
    sample = np.random.choice(X, size=n, replace=True)
    mad_boot.append(mad(sample))
    var_boot.append(variance(sample))

mad_low, mad_high = np.percentile(mad_boot, [2.5, 97.5])
var_low, var_high = np.percentile(var_boot, [2.5, 97.5])

mad_est = mad(X)
var_est = variance(X)

median_true = np.log(2)
t = (1 + np.sqrt(5)) / 2
mad_true = np.log(t)
var_true = 1.0

print(f"MAD:       оценка = {mad_est:.4f}, 95% ДИ = [{mad_low:.4f}; {mad_high:.4f}], "
      f"истинное = {mad_true:.4f}")
print(f"Дисперсия: оценка = {var_est:.4f}, 95% ДИ = [{var_low:.4f}; {var_high:.4f}], "
      f"истинное = {var_true:.4f}")

mad_rel = (mad_high - mad_low) / mad_est * 100
var_rel = (var_high - var_low) / var_est * 100
print(f"\nОтносительная ширина ДИ для MAD:       {mad_rel:.3f}")
print(f"Относительная ширина ДИ для дисперсии: {var_rel:.3f}")
