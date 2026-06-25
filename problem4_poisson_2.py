# -*- coding: utf-8 -*-
"""
第四题：Poisson 群体编码的最大似然估计数值验证

本代码与 TeX 正文中的参数保持一致：
    N = 60, T_obs = 50, s_true = 72, sigma_a = 20, 重复模拟 1000 次。

模型：
    K_a ~ Poisson(T_obs f_a(s))
    f_a(s) = exp(-(s-s_a)^2/(2 sigma_a^2))

估计：
    用网格搜索计算最大似然估计 s_hat。
"""

import numpy as np
import matplotlib.pyplot as plt


# =========================
# 1. 参数设置
# =========================
N = 60
T_obs = 50.0
s_true = 72.0
sigma = 20.0
n_trials = 1000
seed = 2026

# s_a 在 [0,180] 上均匀分布
s_a = np.linspace(0.0, 180.0, N)
sigma_a = np.full(N, sigma)

# 用 0.01 度网格搜索 MLE
grid = np.linspace(0.0, 180.0, 18001)


# =========================
# 2. 基本函数
# =========================
def tuning_curve(u):
    """返回所有神经元在方向 u 下的调谐曲线值。"""
    return np.exp(-((u - s_a) ** 2) / (2.0 * sigma_a ** 2))


def fisher_information(s):
    """Fisher 信息 I(s)。"""
    f = tuning_curve(s)
    return T_obs * np.sum(f * ((s - s_a) ** 2) / (sigma_a ** 4))


def mle_grid_batch(counts, batch_size=250):
    """对一批 Poisson 计数做网格搜索 MLE。"""
    F_grid = np.exp(
        -((grid[:, None] - s_a[None, :]) ** 2) / (2.0 * sigma_a[None, :] ** 2)
    )
    logF_grid = np.log(F_grid)
    sumF_grid = F_grid.sum(axis=1)

    estimates = []
    for i in range(0, counts.shape[0], batch_size):
        c = counts[i:i + batch_size]
        log_likelihood = c @ logF_grid.T - T_obs * sumF_grid[None, :]
        estimates.append(grid[np.argmax(log_likelihood, axis=1)])

    return np.concatenate(estimates)


# =========================
# 3. 数值模拟
# =========================
rng = np.random.default_rng(seed)

rates = tuning_curve(s_true)
poisson_means = T_obs * rates

counts = rng.poisson(poisson_means, size=(n_trials, N)).astype(float)
estimates = mle_grid_batch(counts)

mean_est = estimates.mean()
mse = np.mean((estimates - s_true) ** 2)

I = fisher_information(s_true)
crlb = 1.0 / I

print("Problem 4: Poisson population coding simulation")
print(f"number of trials       = {n_trials}")
print(f"mean of MLE estimates  = {mean_est:.6f}")
print(f"empirical MSE          = {mse:.6f}")
print(f"1 / Fisher information = {crlb:.6f}")


# =========================
# 4. 保存图像，用于插入 TeX
# =========================
fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.hist(estimates, bins=30, alpha=0.75, label=r"MLE estimates")
ax.axvline(s_true, linestyle="--", label=r"true $s=72$")
ax.set_xlabel(r"$\widehat{s}$")
ax.set_ylabel("frequency")
ax.set_title("Problem 4: Poisson-MLE simulation")
ax.legend()

text = (
    f"number of trials = {n_trials}\n"
    f"mean of MLE estimates = {mean_est:.6f}\n"
    f"empirical MSE = {mse:.6f}\n"
    f"1 / Fisher information = {crlb:.6f}"
)
ax.text(0.03, 0.97, text, transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round", alpha=0.15))

plt.tight_layout()
plt.savefig("image-4.png", dpi=150)
plt.close()
