#!/usr/bin/env python
"""
Slice-selective RF pulse simulation for MRI.

This script designs a Hamming-windowed sinc RF pulse, applies it with a
constant slice-selection gradient, and integrates the rotating-frame Bloch
equations without relaxation:

    dM/dt = gamma * M x Beff

The result is a spatial excitation profile across z.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from scipy.fft import fft, fftfreq, fftshift


@dataclass
class SimulationParams:
    """Physical and numerical parameters for the RF pulse simulation."""

    gamma: float = 2 * np.pi * 42.577e6  # rad/s/T for 1H
    duration: float = 4.0e-3  # seconds
    gradient_gz: float = 20.0e-3  # T/m
    slice_thickness: float = 5.0e-3  # m, approximate target full-width slice
    flip_angle_deg: float = 90.0
    dt: float = 2.0e-6  # seconds
    z_min: float = -10.0e-3
    z_max: float = 10.0e-3
    n_z: int = 401
    time_bandwidth: float | None = None


def make_sinc_rf_pulse(
    duration: float,
    dt: float,
    flip_angle_rad: float,
    gamma: float,
    time_bandwidth: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a Hamming-windowed sinc RF pulse and normalize it to a flip angle.

    In the rotating frame, an on-resonance spin at z=0 sees only transverse B1.
    Ignoring relaxation and off-resonance, the flip angle is approximately:

        alpha = gamma * integral(B1(t) dt)

    We scale the pulse so this integral equals the requested flip angle.
    """

    n_t = int(np.round(duration / dt)) + 1
    t = np.linspace(-duration / 2, duration / 2, n_t)

    # np.sinc(x) is sin(pi*x)/(pi*x). A TBW-lobed sinc gives a controlled RF
    # excitation bandwidth. Multiplying by a Hamming window reduces ringing in
    # the slice profile at the cost of a slightly wider transition band.
    sinc_arg = time_bandwidth * t / duration
    rf_shape = np.sinc(sinc_arg)
    rf_shape *= np.hamming(n_t)

    # Scale the signed RF area to produce the desired center flip angle.
    area = np.trapz(rf_shape, t)
    b1 = rf_shape * (flip_angle_rad / (gamma * area))

    # Return time starting at zero for nicer plotting/integration.
    return t + duration / 2, b1


def bloch_step(
    magnetization: np.ndarray,
    b1_t: float,
    delta_omega: float | np.ndarray,
    gamma: float,
    dt: float,
) -> np.ndarray:
    """
    Advance magnetization by one time step using Rodrigues' rotation formula.

    The effective field in the rotating frame is:

        Beff = [B1(t), 0, delta_omega/gamma]

    where delta_omega is the position-dependent off-resonance created by the
    slice-selection gradient. During a small dt, Beff is treated as constant.
    """

    magnetization = np.asarray(magnetization, dtype=float)
    delta_omega = np.asarray(delta_omega, dtype=float)

    # Support both a single spin, shaped (3,), and many spins, shaped (N, 3).
    if magnetization.ndim == 1:
        omega_vec = np.array([gamma * b1_t, 0.0, float(delta_omega)], dtype=float)
        omega = np.linalg.norm(omega_vec)

        if omega == 0.0:
            return magnetization

        axis = omega_vec / omega
        theta = omega * dt

        # dM/dt = gamma M x B rotates M by -theta about Beff under this
        # convention.
        return (
            magnetization * np.cos(theta)
            - np.cross(axis, magnetization) * np.sin(theta)
            + axis * np.dot(axis, magnetization) * (1.0 - np.cos(theta))
        )

    omega_vec = np.column_stack(
        [
            np.full(delta_omega.shape, gamma * b1_t),
            np.zeros_like(delta_omega),
            delta_omega,
        ]
    )
    omega = np.linalg.norm(omega_vec, axis=1)
    axis = np.divide(
        omega_vec,
        omega[:, None],
        out=np.zeros_like(omega_vec),
        where=omega[:, None] != 0.0,
    )
    theta = omega * dt
    dot = np.sum(axis * magnetization, axis=1)

    return (
        magnetization * np.cos(theta)[:, None]
        - np.cross(axis, magnetization) * np.sin(theta)[:, None]
        + axis * dot[:, None] * (1.0 - np.cos(theta))[:, None]
    )


def simulate_position(
    z: float,
    t: np.ndarray,
    b1: np.ndarray,
    gamma: float,
    gradient_gz: float,
) -> np.ndarray:
    """
    Simulate one spin located at position z.

    The slice-selection gradient makes the Larmor frequency depend on z:

        delta_omega(z) = gamma * Gz * z

    Spins near the RF pulse bandwidth are tipped into the transverse plane;
    spins far away remain close to +Mz.
    """

    dt = float(t[1] - t[0])
    delta_omega = gamma * gradient_gz * z
    magnetization = np.array([0.0, 0.0, 1.0], dtype=float)

    for b1_t in b1:
        magnetization = bloch_step(magnetization, b1_t, delta_omega, gamma, dt)

    return magnetization


def simulate_positions(
    z: np.ndarray,
    t: np.ndarray,
    b1: np.ndarray,
    gamma: float,
    gradient_gz: float,
) -> np.ndarray:
    """Vectorized version of simulate_position for the full spatial profile."""

    dt = float(t[1] - t[0])
    delta_omega = gamma * gradient_gz * z
    magnetization = np.zeros((len(z), 3), dtype=float)
    magnetization[:, 2] = 1.0

    for b1_t in b1:
        magnetization = bloch_step(magnetization, b1_t, delta_omega, gamma, dt)

    return magnetization


def run_simulation(params: SimulationParams) -> dict[str, np.ndarray | float]:
    """Create the RF pulse, simulate all z positions, and compute diagnostics."""

    gamma_hz_per_t = params.gamma / (2 * np.pi)

    # For a slice-selective pulse, RF bandwidth approximately maps to spatial
    # width through bandwidth_hz = gamma_bar * Gz * slice_thickness.
    # TBW = bandwidth_hz * pulse_duration.
    if params.time_bandwidth is None:
        bandwidth_hz = gamma_hz_per_t * params.gradient_gz * params.slice_thickness
        time_bandwidth = bandwidth_hz * params.duration
    else:
        time_bandwidth = params.time_bandwidth

    flip_angle_rad = np.deg2rad(params.flip_angle_deg)
    t, b1 = make_sinc_rf_pulse(
        params.duration,
        params.dt,
        flip_angle_rad,
        params.gamma,
        time_bandwidth,
    )

    z = np.linspace(params.z_min, params.z_max, params.n_z)
    final_m = simulate_positions(z, t, b1, params.gamma, params.gradient_gz)

    transverse = np.hypot(final_m[:, 0], final_m[:, 1])
    flip_angle = np.rad2deg(np.arccos(np.clip(final_m[:, 2], -1.0, 1.0)))

    # FFT of the RF waveform gives the pulse's excitation bandwidth intuition.
    spectrum = fftshift(np.abs(fft(b1)))
    freqs_hz = fftshift(fftfreq(len(b1), d=params.dt))

    return {
        "t": t,
        "b1": b1,
        "z": z,
        "final_m": final_m,
        "transverse": transverse,
        "flip_angle": flip_angle,
        "freqs_hz": freqs_hz,
        "spectrum": spectrum / np.max(spectrum),
        "time_bandwidth": time_bandwidth,
        "estimated_bandwidth_hz": time_bandwidth / params.duration,
    }


def plot_results(
    results: dict[str, np.ndarray | float],
    params: SimulationParams,
    save_path: str | None = None,
) -> tuple[plt.Figure, np.ndarray]:
    """Plot RF waveform, spectrum, final magnetization, and flip angle profile."""

    t_ms = np.asarray(results["t"]) * 1e3
    b1_ut = np.asarray(results["b1"]) * 1e6
    z_mm = np.asarray(results["z"]) * 1e3
    final_m = np.asarray(results["final_m"])
    freqs_khz = np.asarray(results["freqs_hz"]) / 1e3
    spectrum = np.asarray(results["spectrum"])
    flip_angle = np.asarray(results["flip_angle"])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    fig.suptitle("Slice-Selective RF Excitation via the Bloch Equations", fontsize=14)

    axes[0, 0].plot(t_ms, b1_ut, color="#0b7285", lw=2)
    axes[0, 0].set_title("Hamming-Windowed Sinc RF Pulse")
    axes[0, 0].set_xlabel("Time (ms)")
    axes[0, 0].set_ylabel("B1 amplitude (uT)")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(freqs_khz, spectrum, color="#5f3dc4", lw=2)
    axes[0, 1].axvline(
        float(results["estimated_bandwidth_hz"]) / 2e3,
        color="black",
        ls="--",
        lw=1,
        alpha=0.45,
    )
    axes[0, 1].axvline(
        -float(results["estimated_bandwidth_hz"]) / 2e3,
        color="black",
        ls="--",
        lw=1,
        alpha=0.45,
    )
    axes[0, 1].set_xlim(-12, 12)
    axes[0, 1].set_title("RF Pulse Spectrum")
    axes[0, 1].set_xlabel("Frequency (kHz)")
    axes[0, 1].set_ylabel("Normalized magnitude")
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].plot(z_mm, final_m[:, 0], label="Mx", lw=2)
    axes[1, 0].plot(z_mm, final_m[:, 1], label="My", lw=2)
    axes[1, 0].plot(z_mm, final_m[:, 2], label="Mz", lw=2)
    axes[1, 0].axvspan(
        -params.slice_thickness * 1e3 / 2,
        params.slice_thickness * 1e3 / 2,
        color="gray",
        alpha=0.12,
        label="target slice",
    )
    axes[1, 0].set_title("Final Magnetization vs Position")
    axes[1, 0].set_xlabel("z position (mm)")
    axes[1, 0].set_ylabel("Magnetization")
    axes[1, 0].set_ylim(-1.1, 1.1)
    axes[1, 0].legend(loc="best")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(z_mm, flip_angle, color="#c92a2a", lw=2)
    axes[1, 1].axhline(params.flip_angle_deg, color="black", ls="--", lw=1, alpha=0.45)
    axes[1, 1].axvspan(
        -params.slice_thickness * 1e3 / 2,
        params.slice_thickness * 1e3 / 2,
        color="gray",
        alpha=0.12,
    )
    axes[1, 1].set_title("Flip Angle Profile")
    axes[1, 1].set_xlabel("z position (mm)")
    axes[1, 1].set_ylabel("Flip angle (degrees)")
    axes[1, 1].set_ylim(0, max(110, 1.1 * np.max(flip_angle)))
    axes[1, 1].grid(True, alpha=0.3)

    subtitle = (
        f"Gz={params.gradient_gz * 1e3:.1f} mT/m, "
        f"duration={params.duration * 1e3:.1f} ms, "
        f"TBW={float(results['time_bandwidth']):.2f}"
    )
    fig.text(0.5, 0.005, subtitle, ha="center", fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=180)

    return fig, axes


def add_interactive_sliders(params: SimulationParams) -> None:
    """
    Optional matplotlib-widget UI for quick design exploration.

    Recomputing the full Bloch simulation is more expensive than simple plotting,
    so the sliders use a moderate number of z positions for responsiveness.
    """

    live_params = SimulationParams(
        gamma=params.gamma,
        duration=params.duration,
        gradient_gz=params.gradient_gz,
        slice_thickness=params.slice_thickness,
        flip_angle_deg=params.flip_angle_deg,
        dt=params.dt,
        z_min=params.z_min,
        z_max=params.z_max,
        n_z=181,
        time_bandwidth=params.time_bandwidth,
    )

    results = run_simulation(live_params)
    fig, axes = plot_results(results, live_params)
    fig.subplots_adjust(bottom=0.25)

    slider_specs = [
        ("Duration (ms)", 1.0, 8.0, live_params.duration * 1e3),
        ("TBW", 2.0, 20.0, float(results["time_bandwidth"])),
        ("Gz (mT/m)", 5.0, 40.0, live_params.gradient_gz * 1e3),
        ("Flip (deg)", 10.0, 180.0, live_params.flip_angle_deg),
    ]
    sliders = []
    for idx, (label, vmin, vmax, value) in enumerate(slider_specs):
        ax = fig.add_axes([0.15, 0.16 - idx * 0.035, 0.72, 0.022])
        sliders.append(Slider(ax, label, vmin, vmax, valinit=value))

    def update(_value: float) -> None:
        live_params.duration = sliders[0].val * 1e-3
        live_params.time_bandwidth = sliders[1].val
        live_params.gradient_gz = sliders[2].val * 1e-3
        live_params.flip_angle_deg = sliders[3].val

        new_results = run_simulation(live_params)

        # Simplicity over cleverness: redraw the current figure contents.
        for ax in axes.flat:
            ax.clear()
        plot_results(new_results, live_params)
        fig.canvas.draw_idle()

    for slider in sliders:
        slider.on_changed(update)

    plt.show()


def print_explanation(results: dict[str, np.ndarray | float], params: SimulationParams) -> None:
    """Print a concise interpretation of the simulation output."""

    bandwidth_hz = float(results["estimated_bandwidth_hz"])
    slice_mm = bandwidth_hz / ((params.gamma / (2 * np.pi)) * params.gradient_gz) * 1e3

    print("\nMRI RF pulse simulation complete.")
    print(
        f"The RF pulse was normalized so the on-resonance center spin receives "
        f"an approximately {params.flip_angle_deg:.0f}-degree excitation."
    )
    print(
        f"The FFT shows the excitation bandwidth of the sinc pulse. With "
        f"Gz={params.gradient_gz * 1e3:.1f} mT/m, that bandwidth maps to an "
        f"estimated slice width of about {slice_mm:.2f} mm."
    )
    print(
        "The magnetization and flip-angle plots show that spins near z=0 are "
        "tipped into the transverse plane, while spins outside the RF bandwidth "
        "remain mostly along +Mz."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate slice-selective RF excitation in MRI."
    )
    parser.add_argument("--save", default=None, help="Optional path for saving the plot PNG.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Enable matplotlib sliders for pulse duration, TBW, gradient, and flip angle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = SimulationParams()

    if args.interactive:
        add_interactive_sliders(params)
        return

    results = run_simulation(params)
    plot_results(results, params, save_path=args.save)
    print_explanation(results, params)
    if matplotlib.get_backend().lower() != "agg":
        plt.show()


if __name__ == "__main__":
    main()
