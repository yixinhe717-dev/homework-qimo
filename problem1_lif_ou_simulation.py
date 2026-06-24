import numpy as np


def simulate_one_isi_lif_ou(
    rng,
    g_L=1.0,
    tau=0.05,
    tau_ref=0.1,
    V_r=0.0,
    V_th=1.0,
    g_syn=1.0,
    mu=1.685118,
    sigma=0.055236,
    dt=1e-4,
    t_max=5.0,
):
    """
    Simulate one inter-spike interval for the LIF-OU model.

    Here V is reset to V_r and I is initialized at mu in each trial.
    This matches the iid inter-spike-interval approximation used in the written solution.
    """
    V = V_r
    I = mu
    t = 0.0
    sqrt_dt = np.sqrt(dt)

    while t < t_max:
        V_old = V

        dW = sqrt_dt * rng.normal()
        I = I + (mu - I) / tau * dt + (sigma / tau) * dW

        V = V + (-g_L * V + g_syn * I) * dt
        t += dt

        if V >= V_th:
            if V != V_old:
                alpha = (V_th - V_old) / (V - V_old)
                hit_time = t - dt + alpha * dt
            else:
                hit_time = t

            return tau_ref + hit_time

    raise RuntimeError("No spike occurred before t_max. Increase t_max.")


def main():
    n_trials = 1000
    seed = 2026
    rng = np.random.default_rng(seed)

    isis = np.array([
        simulate_one_isi_lif_ou(rng)
        for _ in range(n_trials)
    ])

    print("Problem 1: LIF-OU numerical simulation")
    print(f"number of trials = {n_trials}")
    print(f"sample mean       = {isis.mean():.6f}")
    print(f"sample variance   = {isis.var(ddof=1):.6f}")


if __name__ == "__main__":
    main()
