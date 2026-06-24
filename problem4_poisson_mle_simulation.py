import numpy as np


def tuning_curve(u, s_a, sigma_a):
    """
    Gaussian tuning curve with peak equal to 1.
    """
    return np.exp(-((u - s_a) ** 2) / (2.0 * sigma_a ** 2))


def mle_direction_grid(K, s_a, sigma_a, T_obs, grid):
    """
    Grid-search MLE for the stimulus direction.
    """
    F = tuning_curve(grid[:, None], s_a[None, :], sigma_a[None, :])
    log_likelihood = K @ np.log(F).T - T_obs * F.sum(axis=1)

    return grid[np.argmax(log_likelihood)]


def fisher_information(s, s_a, sigma_a, T_obs):
    f = tuning_curve(s, s_a, sigma_a)
    return T_obs * np.sum(f * ((s - s_a) ** 2) / (sigma_a ** 4))


def main():
    n_trials = 1000
    seed = 2026
    rng = np.random.default_rng(seed)

    N = 60
    s_true = 72.0
    T_obs = 50.0

    s_a = np.linspace(0.0, 180.0, N)
    sigma_a = np.full(N, 20.0)
    grid = np.linspace(0.0, 180.0, 18001)

    rates = tuning_curve(s_true, s_a, sigma_a)
    poisson_means = T_obs * rates

    estimates = []

    for _ in range(n_trials):
        K = rng.poisson(poisson_means)
        s_hat = mle_direction_grid(K, s_a, sigma_a, T_obs, grid)
        estimates.append(s_hat)

    estimates = np.array(estimates)
    mse = np.mean((estimates - s_true) ** 2)

    I = fisher_information(s_true, s_a, sigma_a, T_obs)
    crlb = 1.0 / I

    print("Problem 4: Poisson population coding simulation")
    print(f"number of trials       = {n_trials}")
    print(f"mean of MLE estimates  = {estimates.mean():.6f}")
    print(f"empirical MSE          = {mse:.6f}")
    print(f"1 / Fisher information = {crlb:.6f}")


if __name__ == "__main__":
    main()
