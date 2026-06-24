import numpy as np


def ei_jacobian(
    tau_E=1.0,
    tau_I=1.0,
    M_EE=3.0,
    M_EI=4.0,
    M_IE=4.0,
    M_II=1.0,
):
    """
    Jacobian matrix in the fully active linear region of the E-I model.
    """
    a = M_EE - 1.0
    b = M_II + 1.0

    J = np.array([
        [a / tau_E, -M_EI / tau_E],
        [M_IE / tau_I, -b / tau_I],
    ])

    return J


def main():
    tau_E = 1.0
    M_EE = 3.0
    M_EI = 4.0
    M_IE = 4.0
    M_II = 1.0

    a = M_EE - 1.0
    b = M_II + 1.0
    tau_I_H = b * tau_E / a

    print("Problem 3: linearized E-I system")
    print(f"critical tau_I = {tau_I_H:.6f}")
    print()

    for tau_I in [0.8, 1.0, 1.2]:
        J = ei_jacobian(
            tau_E=tau_E,
            tau_I=tau_I,
            M_EE=M_EE,
            M_EI=M_EI,
            M_IE=M_IE,
            M_II=M_II,
        )

        eigvals = np.linalg.eigvals(J)

        print(f"tau_I = {tau_I:.1f}")
        print("Jacobian:")
        print(J)

        for lam in eigvals:
            print(f"  lambda = {lam.real:.6f} {lam.imag:+.6f} i")

        print()


if __name__ == "__main__":
    main()
