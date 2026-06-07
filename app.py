#!/usr/bin/env python
"""
Interactive MRI slice-selective RF pulse simulator.

Run with:

    streamlit run app.py

The app reuses the Bloch-equation simulation functions from simulation.py and
adds browser-based controls for RF design.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ModuleNotFoundError:
    go = None
    make_subplots = None

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from simulation import SimulationParams, run_simulation


PRESETS = {
    "Balanced 5 mm excitation": {
        "values": (90, 4.0, 20.0, 5.0, None),
        "description": (
            "A general-purpose 90 degree slice-selective excitation. It targets a "
            "moderate 5 mm slice using a 4 ms RF pulse and a 20 mT/m gradient."
        ),
    },
    "Thin 2 mm high-resolution slice": {
        "values": (90, 6.0, 28.0, 2.0, None),
        "description": (
            "A higher-resolution slice prescription. It uses a stronger gradient "
            "and longer RF pulse to keep the selected slab narrow."
        ),
    },
    "Thick 10 mm localizer slice": {
        "values": (70, 2.4, 16.0, 10.0, None),
        "description": (
            "A quick, broad scout-style excitation where coverage and speed matter "
            "more than a sharp slice edge."
        ),
    },
    "Sharp 3 mm TBW-optimized slice": {
        "values": (90, 7.0, 32.0, 3.0, 14.0),
        "description": (
            "A sharper profile with a higher time-bandwidth product. This emphasizes "
            "the tradeoff between cleaner slice edges and pulse complexity."
        ),
    },
    "Low-gradient 8 mm body coil scenario": {
        "values": (90, 5.5, 10.0, 8.0, None),
        "description": (
            "A lower-gradient prescription that needs a narrower RF bandwidth for "
            "the same anatomical coverage."
        ),
    },
    "Broad 180-degree inversion": {
        "values": (180, 5.0, 12.0, 10.0, 8.0),
        "description": (
            "A wide inversion-style pulse. The center spin is targeted for 180 "
            "degrees, flipping magnetization from +Mz toward -Mz."
        ),
    },
    "Manual": {
        "values": (90, 4.0, 20.0, 5.0, None),
        "description": (
            "Neutral starting values. Use this when you want to explore the design "
            "space without assuming a particular imaging goal."
        ),
    },
}


COMPARISON_PRESETS = (
    "Thin 2 mm high-resolution slice",
    "Balanced 5 mm excitation",
    "Thick 10 mm localizer slice",
    "Sharp 3 mm TBW-optimized slice",
    "Broad 180-degree inversion",
)


st.set_page_config(
    page_title="MRI RF Pulse Designer",
    page_icon="MRI",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_style() -> None:
    """Apply a restrained technical dashboard style."""

    st.markdown(
        """
        <style>
        :root {
            --rf-ink: #14213d;
            --rf-muted: #5c667a;
            --rf-cyan: #087f8c;
            --rf-violet: #6741d9;
            --rf-amber: #f08c00;
            --rf-red: #d9480f;
            --rf-panel: #ffffff;
            --rf-line: #d8dee9;
            --rf-soft: #f6f8fb;
        }

        .stApp {
            background:
                linear-gradient(180deg, #f8fafc 0%, #eef3f7 48%, #f7f8fb 100%);
            color: var(--rf-ink);
        }

        section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid var(--rf-line);
        }

        .hero {
            border: 1px solid rgba(20, 33, 61, 0.14);
            background:
                radial-gradient(circle at 13% 16%, rgba(8, 127, 140, 0.13), transparent 30%),
                radial-gradient(circle at 86% 10%, rgba(240, 140, 0, 0.14), transparent 25%),
                linear-gradient(135deg, #ffffff 0%, #eef5f7 58%, #f9fbfd 100%);
            border-radius: 8px;
            padding: 26px 28px;
            margin: 0 0 18px 0;
            box-shadow: 0 12px 38px rgba(20, 33, 61, 0.08);
        }

        .hero h1 {
            font-size: 2.35rem;
            line-height: 1.08;
            margin: 0 0 10px 0;
            letter-spacing: 0;
            color: var(--rf-ink);
        }

        .hero p {
            max-width: 820px;
            margin: 0;
            color: var(--rf-muted);
            font-size: 1.02rem;
        }

        .badge-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 14px;
        }

        .badge {
            border: 1px solid rgba(8, 127, 140, 0.25);
            background: rgba(255, 255, 255, 0.72);
            color: var(--rf-ink);
            border-radius: 999px;
            padding: 5px 10px;
            font-size: 0.78rem;
            font-weight: 650;
        }

        .metric-card {
            border: 1px solid rgba(20, 33, 61, 0.12);
            background: var(--rf-panel);
            border-radius: 8px;
            padding: 14px 15px;
            box-shadow: 0 8px 26px rgba(20, 33, 61, 0.06);
            min-height: 94px;
        }

        .metric-label {
            color: var(--rf-muted);
            font-size: 0.78rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0;
        }

        .metric-value {
            color: var(--rf-ink);
            font-size: 1.55rem;
            font-weight: 760;
            margin-top: 4px;
        }

        .metric-note {
            color: var(--rf-muted);
            font-size: 0.82rem;
            margin-top: 2px;
        }

        .design-verdict {
            border-left: 4px solid var(--rf-cyan);
            background: #ffffff;
            border-radius: 8px;
            padding: 13px 15px;
            margin: 8px 0 14px 0;
            color: var(--rf-ink);
            box-shadow: 0 8px 24px rgba(20, 33, 61, 0.05);
        }

        .explain-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 12px;
            margin: 10px 0 16px 0;
        }

        .explain-card {
            border: 1px solid rgba(20, 33, 61, 0.12);
            background: #ffffff;
            border-radius: 8px;
            padding: 15px 16px;
            box-shadow: 0 8px 24px rgba(20, 33, 61, 0.05);
        }

        .explain-card h3 {
            margin: 0 0 7px 0;
            font-size: 1rem;
            color: var(--rf-ink);
        }

        .explain-card p {
            margin: 0;
            color: var(--rf-muted);
            font-size: 0.92rem;
            line-height: 1.42;
        }

        .workflow-step {
            border-left: 3px solid var(--rf-amber);
            background: #ffffff;
            border-radius: 8px;
            padding: 13px 15px;
            margin: 10px 0;
        }

        .workflow-step strong {
            color: var(--rf-ink);
        }

        .small-table {
            border-collapse: collapse;
            width: 100%;
            background: #ffffff;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 8px 24px rgba(20, 33, 61, 0.05);
        }

        .small-table th, .small-table td {
            border-bottom: 1px solid #e8edf3;
            padding: 10px 12px;
            text-align: left;
            vertical-align: top;
            font-size: 0.9rem;
        }

        .small-table th {
            background: #f2f6fa;
            color: var(--rf-ink);
        }

        @media (max-width: 900px) {
            .explain-grid {
                grid-template-columns: 1fr;
            }
        }

        div[data-testid="stMetric"] {
            background: #ffffff;
            border: 1px solid rgba(20, 33, 61, 0.12);
            border-radius: 8px;
            padding: 12px 14px;
        }

        div[data-testid="stPlotlyChart"] {
            border: 1px solid rgba(20, 33, 61, 0.10);
            border-radius: 8px;
            background: #ffffff;
            padding: 4px;
            box-shadow: 0 10px 30px rgba(20, 33, 61, 0.06);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero">
            <div class="badge-row">
                <span class="badge">Bloch equations</span>
                <span class="badge">Windowed sinc RF</span>
                <span class="badge">Slice-select gradient</span>
                <span class="badge">No relaxation</span>
            </div>
            <h1>MRI Slice-Selective RF Pulse Designer</h1>
            <p>
                Tune an excitation pulse and watch its frequency bandwidth map
                into a spatial slice through the gradient field.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_params() -> SimulationParams:
    """Read UI controls and convert them into simulation parameters."""

    st.sidebar.header("RF Design")

    preset = st.sidebar.selectbox(
        "Preset",
        tuple(PRESETS.keys()),
        help="Start from a realistic RF design and then fine tune the sliders.",
    )

    st.sidebar.info(PRESETS[preset]["description"])

    default_flip, default_duration, default_gradient, default_slice, default_tbw = PRESETS[preset]["values"]

    flip_angle_deg = st.sidebar.slider(
        "Target flip angle (degrees)",
        min_value=10,
        max_value=180,
        value=default_flip,
        step=5,
        help=(
            "The intended rotation angle for the on-resonance spin at z=0. "
            "90 degrees creates transverse signal; 180 degrees inverts Mz."
        ),
    )
    duration_ms = st.sidebar.slider(
        "RF pulse duration (ms)",
        min_value=1.0,
        max_value=10.0,
        value=default_duration,
        step=0.25,
        help=(
            "How long the RF pulse is applied. Longer pulses usually require less "
            "peak B1 and can produce narrower frequency bandwidth for a fixed TBW."
        ),
    )
    gradient_mt_m = st.sidebar.slider(
        "Slice-selection gradient Gz (mT/m)",
        min_value=5.0,
        max_value=50.0,
        value=default_gradient,
        step=1.0,
        help=(
            "Gradient field strength during the RF pulse. It converts position "
            "along z into resonance frequency. Stronger Gz makes a given RF "
            "bandwidth select a thinner slice."
        ),
    )
    slice_thickness_mm = st.sidebar.slider(
        "Target slice thickness (mm)",
        min_value=1.0,
        max_value=12.0,
        value=default_slice,
        step=0.5,
        help=(
            "Desired spatial width of the excited slab. When manual TBW is off, "
            "the app computes the RF bandwidth needed to approximate this slice."
        ),
    )

    manual_tbw = st.sidebar.toggle(
        "Set time-bandwidth product manually",
        value=default_tbw is not None,
        help="TBW controls how many sinc lobes fit into the pulse and sets RF bandwidth.",
    )
    time_bandwidth = None
    if manual_tbw:
        time_bandwidth = st.sidebar.slider(
            "Time-bandwidth product (unitless)",
            min_value=2.0,
            max_value=24.0,
            value=float(default_tbw or 8.0),
            step=0.5,
            help=(
                "TBW = RF bandwidth x pulse duration. Higher TBW gives a sharper "
                "slice profile but tends to require more pulse complexity."
            ),
        )

    with st.sidebar.expander("Numerics", expanded=False):
        n_z = st.slider(
            "Spatial samples (count)",
            min_value=101,
            max_value=801,
            value=301,
            step=50,
            help="More positions give a smoother profile but take longer.",
        )
        dt_us = st.slider(
            "Time step (microseconds)",
            min_value=1.0,
            max_value=8.0,
            value=2.0,
            step=0.5,
            help=(
                "Bloch integration step size. Smaller values are more accurate, "
                "but the simulation takes longer."
            ),
        )
        z_range_mm = st.slider(
            "Displayed z range (+/- mm)",
            min_value=5.0,
            max_value=25.0,
            value=10.0,
            step=1.0,
            help="How far from the slice center to simulate and display.",
        )

    st.sidebar.divider()
    with st.sidebar.expander("Control glossary", expanded=True):
        st.markdown(
            """
            - **Flip angle (degrees):** how far magnetization rotates away from the z-axis.
            - **RF duration (ms):** length of the B1 pulse.
            - **Gz (mT/m):** slice-selection gradient strength.
            - **Slice thickness (mm):** target width of the excited slab.
            - **TBW:** unitless sharpness/bandwidth control for the sinc pulse.
            """
        )
    st.sidebar.caption(
        "Tip: thinner slices usually need more gradient, more RF bandwidth, "
        "or a longer pulse."
    )

    return SimulationParams(
        duration=duration_ms * 1e-3,
        gradient_gz=gradient_mt_m * 1e-3,
        slice_thickness=slice_thickness_mm * 1e-3,
        flip_angle_deg=float(flip_angle_deg),
        dt=dt_us * 1e-6,
        z_min=-z_range_mm * 1e-3,
        z_max=z_range_mm * 1e-3,
        n_z=int(n_z),
        time_bandwidth=time_bandwidth,
    )


def calculate_design_summary(
    results: dict[str, np.ndarray | float],
    params: SimulationParams,
) -> dict[str, float | str]:
    """Compute user-facing performance measures for the current pulse."""

    gamma_hz_per_t = params.gamma / (2 * np.pi)
    bandwidth_hz = float(results["estimated_bandwidth_hz"])
    estimated_slice_mm = bandwidth_hz / (gamma_hz_per_t * params.gradient_gz) * 1e3
    peak_b1_ut = float(np.max(np.abs(np.asarray(results["b1"])))) * 1e6
    z = np.asarray(results["z"])
    flip_angle = np.asarray(results["flip_angle"])
    final_m = np.asarray(results["final_m"])
    transverse = np.hypot(final_m[:, 0], final_m[:, 1])
    center_idx = len(z) // 2
    center_flip = float(flip_angle[center_idx])

    in_slice = np.abs(z) <= params.slice_thickness / 2
    out_slice = np.abs(z) >= params.slice_thickness
    passband_ripple = float(np.ptp(flip_angle[in_slice])) if np.any(in_slice) else 0.0
    stopband_flip = float(np.max(flip_angle[out_slice])) if np.any(out_slice) else 0.0
    mean_transverse = float(np.mean(transverse[in_slice])) if np.any(in_slice) else 0.0

    if abs(center_flip - params.flip_angle_deg) < 3 and stopband_flip < 10:
        verdict = "Clean excitation profile"
    elif stopband_flip >= 20:
        verdict = "Noticeable outside-slice excitation"
    elif passband_ripple > 18:
        verdict = "Passband ripple is visible"
    else:
        verdict = "Usable pulse with tradeoffs"

    return {
        "peak_b1_ut": peak_b1_ut,
        "bandwidth_hz": bandwidth_hz,
        "estimated_slice_mm": estimated_slice_mm,
        "center_flip": center_flip,
        "passband_ripple": passband_ripple,
        "stopband_flip": stopband_flip,
        "mean_transverse": mean_transverse,
        "verdict": verdict,
    }


@st.cache_data(show_spinner=False)
def cached_run_simulation(
    duration: float,
    gradient_gz: float,
    slice_thickness: float,
    flip_angle_deg: float,
    dt: float,
    z_min: float,
    z_max: float,
    n_z: int,
    time_bandwidth: float | None,
) -> dict[str, np.ndarray | float]:
    """Cache expensive Bloch simulations for repeated slider states."""

    params = SimulationParams(
        duration=duration,
        gradient_gz=gradient_gz,
        slice_thickness=slice_thickness,
        flip_angle_deg=flip_angle_deg,
        dt=dt,
        z_min=z_min,
        z_max=z_max,
        n_z=n_z,
        time_bandwidth=time_bandwidth,
    )
    return run_simulation(params)


def run_from_params(params: SimulationParams) -> dict[str, np.ndarray | float]:
    return cached_run_simulation(
        params.duration,
        params.gradient_gz,
        params.slice_thickness,
        params.flip_angle_deg,
        params.dt,
        params.z_min,
        params.z_max,
        params.n_z,
        params.time_bandwidth,
    )


def params_from_preset(name: str, base: SimulationParams) -> SimulationParams:
    """Build a comparable simulation from one preset using current numerics."""

    flip_deg, duration_ms, gradient_mt_m, slice_mm, tbw = PRESETS[name]["values"]
    return SimulationParams(
        duration=duration_ms * 1e-3,
        gradient_gz=gradient_mt_m * 1e-3,
        slice_thickness=slice_mm * 1e-3,
        flip_angle_deg=float(flip_deg),
        dt=base.dt,
        z_min=base.z_min,
        z_max=base.z_max,
        n_z=base.n_z,
        time_bandwidth=tbw,
    )


def render_slice_preset_comparison(base_params: SimulationParams) -> None:
    """Compare Bloch slice profiles for scanner-style presets."""

    st.markdown("### Bloch Slice Preset Comparison")
    st.markdown(
        """
        This view reruns the Bloch simulation for several slice prescriptions using
        the same spatial grid. It makes the tradeoff visible: thin slices need more
        bandwidth or gradient strength, thick localizer slices are faster, and
        inversion pulses change the final longitudinal magnetization.
        """
    )

    selected = st.multiselect(
        "Slice prescriptions to overlay",
        COMPARISON_PRESETS,
        default=list(COMPARISON_PRESETS[:4]),
    )
    if not selected:
        st.info("Choose at least one preset to draw the comparison.")
        return

    rows = []
    comparison = {}
    for name in selected:
        params = params_from_preset(name, base_params)
        results = run_from_params(params)
        summary = calculate_design_summary(results, params)
        z_mm = np.asarray(results["z"]) * 1e3
        flip_angle = np.asarray(results["flip_angle"])
        final_m = np.asarray(results["final_m"])
        comparison[name] = (z_mm, flip_angle, final_m[:, 2])
        rows.append(
            {
                "Preset": name,
                "Flip angle deg": params.flip_angle_deg,
                "Slice mm": params.slice_thickness * 1e3,
                "Gz mT/m": params.gradient_gz * 1e3,
                "RF ms": params.duration * 1e3,
                "TBW": float(results["time_bandwidth"]),
                "Peak B1 uT": summary["peak_b1_ut"],
                "Estimated slice mm": summary["estimated_slice_mm"],
                "Stopband max deg": summary["stopband_flip"],
            }
        )

    if go is not None:
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Flip angle across position", "Final Mz across position"),
            horizontal_spacing=0.09,
        )
        palette = ["#087f8c", "#6741d9", "#d9480f", "#2b8a3e", "#f08c00"]
        for i, (name, (z_mm, flip_angle, mz)) in enumerate(comparison.items()):
            color = palette[i % len(palette)]
            fig.add_trace(
                go.Scatter(x=z_mm, y=flip_angle, mode="lines", name=name, line=dict(color=color, width=3)),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(x=z_mm, y=mz, mode="lines", name=name, line=dict(color=color, width=3), showlegend=False),
                row=1,
                col=2,
            )
        fig.update_xaxes(title_text="z position (mm)", row=1, col=1)
        fig.update_xaxes(title_text="z position (mm)", row=1, col=2)
        fig.update_yaxes(title_text="Flip angle (degrees)", row=1, col=1)
        fig.update_yaxes(title_text="Mz", range=[-1.1, 1.1], row=1, col=2)
        fig.update_layout(
            height=560,
            margin=dict(l=20, r=20, t=60, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.04, xanchor="right", x=1),
            paper_bgcolor="#ffffff",
            plot_bgcolor="#fbfcfe",
            font=dict(color="#14213d", family="Inter, Segoe UI, Arial, sans-serif"),
        )
        fig.update_xaxes(gridcolor="#e8edf3", zerolinecolor="#cad2dd")
        fig.update_yaxes(gridcolor="#e8edf3", zerolinecolor="#cad2dd")
        st.plotly_chart(fig, use_container_width=True)
    else:
        z_index = next(iter(comparison.values()))[0]
        flip_df = pd.DataFrame({name: data[1] for name, data in comparison.items()}, index=z_index)
        st.line_chart(flip_df)

    st.dataframe(pd.DataFrame(rows).round(3), width="stretch", hide_index=True)


def make_main_figure(results: dict[str, np.ndarray | float], params: SimulationParams):
    """Create a four-panel interactive Plotly figure."""

    if go is None or make_subplots is None:
        return None

    t_ms = np.asarray(results["t"]) * 1e3
    b1_ut = np.asarray(results["b1"]) * 1e6
    z_mm = np.asarray(results["z"]) * 1e3
    final_m = np.asarray(results["final_m"])
    freqs_khz = np.asarray(results["freqs_hz"]) / 1e3
    spectrum = np.asarray(results["spectrum"])
    flip_angle = np.asarray(results["flip_angle"])

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "RF pulse amplitude",
            "RF pulse spectrum",
            "Final magnetization",
            "Flip angle profile",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )

    fig.add_trace(
        go.Scatter(
            x=t_ms,
            y=b1_ut,
            mode="lines",
            name="B1",
            fill="tozeroy",
            fillcolor="rgba(8, 127, 140, 0.18)",
            line=dict(color="#087f8c", width=3),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=freqs_khz,
            y=spectrum,
            mode="lines",
            name="RF spectrum",
            fill="tozeroy",
            fillcolor="rgba(103, 65, 217, 0.16)",
            line=dict(color="#6741d9", width=3),
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(x=z_mm, y=final_m[:, 0], mode="lines", name="Mx", line=dict(color="#087f8c", width=2)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=z_mm, y=final_m[:, 1], mode="lines", name="My", line=dict(color="#6741d9", width=2)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(x=z_mm, y=final_m[:, 2], mode="lines", name="Mz", line=dict(color="#2b8a3e", width=3)),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=z_mm,
            y=flip_angle,
            mode="lines",
            name="Flip angle",
            fill="tozeroy",
            fillcolor="rgba(217, 72, 15, 0.14)",
            line=dict(color="#d9480f", width=3),
        ),
        row=2,
        col=2,
    )

    slice_half_mm = params.slice_thickness * 1e3 / 2
    bandwidth_half_khz = float(results["estimated_bandwidth_hz"]) / 2e3

    fig.add_vrect(
        x0=-slice_half_mm,
        x1=slice_half_mm,
        fillcolor="rgba(120,120,120,0.14)",
        line_width=0,
        row=2,
        col=1,
    )
    fig.add_vrect(
        x0=-slice_half_mm,
        x1=slice_half_mm,
        fillcolor="rgba(120,120,120,0.14)",
        line_width=0,
        row=2,
        col=2,
    )
    fig.add_vline(x=bandwidth_half_khz, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_vline(x=-bandwidth_half_khz, line_dash="dash", line_color="gray", row=1, col=2)
    fig.add_hline(
        y=params.flip_angle_deg,
        line_dash="dash",
        line_color="gray",
        row=2,
        col=2,
    )

    fig.update_xaxes(title_text="Time (ms)", row=1, col=1)
    fig.update_yaxes(title_text="B1 (uT)", row=1, col=1)
    fig.update_xaxes(title_text="Frequency (kHz)", range=[-12, 12], row=1, col=2)
    fig.update_yaxes(title_text="Normalized magnitude", row=1, col=2)
    fig.update_xaxes(title_text="z position (mm)", row=2, col=1)
    fig.update_yaxes(title_text="Magnetization", range=[-1.1, 1.1], row=2, col=1)
    fig.update_xaxes(title_text="z position (mm)", row=2, col=2)
    fig.update_yaxes(title_text="Flip angle (degrees)", range=[0, max(110, 1.12 * np.max(flip_angle))], row=2, col=2)

    fig.update_layout(
        height=780,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#fbfcfe",
        font=dict(color="#14213d", family="Inter, Segoe UI, Arial, sans-serif"),
    )
    fig.update_xaxes(gridcolor="#e8edf3", zerolinecolor="#cad2dd")
    fig.update_yaxes(gridcolor="#e8edf3", zerolinecolor="#cad2dd")
    return fig


def render_fallback_charts(results: dict[str, np.ndarray | float], params: SimulationParams) -> None:
    """Render built-in Streamlit charts when Plotly is not installed."""

    st.warning(
        "Plotly is not installed in this Python environment, so the app is using "
        "Streamlit's built-in charts. Install Plotly for hoverable four-panel plots."
    )

    t_ms = np.asarray(results["t"]) * 1e3
    b1_ut = np.asarray(results["b1"]) * 1e6
    z_mm = np.asarray(results["z"]) * 1e3
    final_m = np.asarray(results["final_m"])
    freqs_khz = np.asarray(results["freqs_hz"]) / 1e3
    spectrum = np.asarray(results["spectrum"])
    flip_angle = np.asarray(results["flip_angle"])

    cols = st.columns(2)
    with cols[0]:
        st.subheader("RF pulse amplitude")
        st.line_chart(pd.DataFrame({"B1 (uT)": b1_ut}, index=t_ms))
    with cols[1]:
        st.subheader("RF pulse spectrum")
        spectrum_df = pd.DataFrame({"Normalized magnitude": spectrum}, index=freqs_khz)
        st.line_chart(spectrum_df[(spectrum_df.index >= -12) & (spectrum_df.index <= 12)])

    cols = st.columns(2)
    with cols[0]:
        st.subheader("Final magnetization")
        st.line_chart(
            pd.DataFrame(
                {"Mx": final_m[:, 0], "My": final_m[:, 1], "Mz": final_m[:, 2]},
                index=z_mm,
            )
        )
        st.caption(
            f"Requested slice region: +/- {params.slice_thickness * 1e3 / 2:.1f} mm."
        )
    with cols[1]:
        st.subheader("Flip angle profile")
        st.line_chart(pd.DataFrame({"Flip angle (deg)": flip_angle}, index=z_mm))
        st.caption(f"Dashed target in the Plotly view: {params.flip_angle_deg:.0f} degrees.")


def render_metrics(results: dict[str, np.ndarray | float], params: SimulationParams) -> None:
    """Show derived quantities that connect RF settings to MRI slice physics."""

    summary = calculate_design_summary(results, params)

    cols = st.columns(4)
    cards = (
        ("Peak B1", f"{summary['peak_b1_ut']:.2f} uT", "RF drive amplitude"),
        ("RF bandwidth", f"{summary['bandwidth_hz'] / 1e3:.2f} kHz", "Frequency passband"),
        ("Estimated slice", f"{summary['estimated_slice_mm']:.2f} mm", "Bandwidth mapped by Gz"),
        ("Center flip", f"{summary['center_flip']:.1f} deg", "On-resonance spin"),
    )
    for col, (label, value, note) in zip(cols, cards):
        col.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="design-verdict">
            <strong>{summary['verdict']}</strong> &nbsp; Passband ripple:
            {summary['passband_ripple']:.1f} deg &nbsp; Stopband max:
            {summary['stopband_flip']:.1f} deg &nbsp; Mean in-slice transverse:
            {summary['mean_transverse']:.2f}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_quick_guide(params: SimulationParams) -> None:
    """Explain the current RF design in plain language."""

    st.markdown(
        f"""
        <div class="design-verdict">
            <strong>Current prescription:</strong> apply a {params.duration * 1e3:.2f} ms
            windowed sinc RF pulse while a {params.gradient_gz * 1e3:.1f} mT/m gradient
            is on. The pulse is scaled so the center of the slice, z = 0 mm, receives
            about {params.flip_angle_deg:.0f} degrees of rotation. The shaded region in
            the plots marks the requested {params.slice_thickness * 1e3:.1f} mm slice.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_mri_context(params: SimulationParams, results: dict[str, np.ndarray | float]) -> None:
    """Explain how the simulated design maps to an MRI operator workflow."""

    summary = calculate_design_summary(results, params)
    tbw = float(results["time_bandwidth"])
    st.markdown("### How This Relates To An Actual MRI Scanner")
    st.markdown(
        f"""
        <div class="explain-grid">
            <div class="explain-card">
                <h3>What the operator sees</h3>
                <p>
                    On a clinical scanner, the technologist usually chooses protocol
                    settings: slice thickness in mm, scan plane, field of view, TR/TE,
                    flip angle, and sequence type. The console rarely asks them to
                    draw this sinc pulse directly.
                </p>
            </div>
            <div class="explain-card">
                <h3>What the sequence designer controls</h3>
                <p>
                    Behind the protocol, the scanner converts those choices into RF
                    waveform amplitude, RF bandwidth, gradient strength, and timing.
                    This simulator shows that hidden RF-and-gradient design layer.
                </p>
            </div>
            <div class="explain-card">
                <h3>Current operator-style prescription</h3>
                <p>
                    Excite a {params.slice_thickness * 1e3:.1f} mm slice with a
                    {params.flip_angle_deg:.0f} degree RF pulse. The scanner would
                    schedule a {params.duration * 1e3:.2f} ms pulse while Gz is
                    {params.gradient_gz * 1e3:.1f} mT/m.
                </p>
            </div>
            <div class="explain-card">
                <h3>Hidden design result</h3>
                <p>
                    The pulse has TBW {tbw:.2f}, RF bandwidth
                    {summary['bandwidth_hz'] / 1e3:.2f} kHz, and peak B1
                    {summary['peak_b1_ut']:.2f} uT. Those hidden quantities produce
                    the simulated slice profile.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Operator Console vs Pulse Design")
    st.markdown(
        """
        <table class="small-table">
            <thead>
                <tr>
                    <th>Clinical MRI console setting</th>
                    <th>What the scanner does underneath</th>
                    <th>Where it appears here</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td>Slice thickness (mm)</td>
                    <td>Chooses a combination of RF bandwidth and slice-selection gradient.</td>
                    <td>Target slice thickness, RF spectrum, flip angle profile.</td>
                </tr>
                <tr>
                    <td>Flip angle (degrees)</td>
                    <td>Scales RF pulse area so the center spin rotates by that amount.</td>
                    <td>Target flip angle, Peak B1, Center flip.</td>
                </tr>
                <tr>
                    <td>Scan plane and slice position</td>
                    <td>Applies the slice gradient along the selected physical axis and shifts RF center frequency.</td>
                    <td>This simplified simulator only models z position from -10 mm to +10 mm.</td>
                </tr>
                <tr>
                    <td>Sequence type</td>
                    <td>Defines RF pulse shape, gradient timing, echo timing, and readout strategy.</td>
                    <td>This app focuses only on the slice-selective excitation pulse.</td>
                </tr>
                <tr>
                    <td>Image contrast choices</td>
                    <td>Uses TR, TE, inversion time, spoiling, and relaxation physics.</td>
                    <td>Relaxation is intentionally off here so slice selection is isolated.</td>
                </tr>
            </tbody>
        </table>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### What An MRI Machine Would Tell The Operator")
    st.markdown(
        f"""
        <div class="workflow-step">
            <strong>1. Pick the anatomy and plane.</strong> The technologist selects,
            for example, axial, sagittal, or coronal slices. In this simplified app,
            we assume the selected direction is z.
        </div>
        <div class="workflow-step">
            <strong>2. Enter slice thickness.</strong> For the current design that is
            {params.slice_thickness * 1e3:.1f} mm. The scanner then chooses RF
            bandwidth and Gz so only that slab is excited.
        </div>
        <div class="workflow-step">
            <strong>3. Choose flip angle and sequence.</strong> A 90 degree pulse is
            common for creating transverse signal; a 180 degree pulse is used for
            inversion or refocusing. This app scales B1 to hit
            {params.flip_angle_deg:.0f} degrees at the slice center.
        </div>
        <div class="workflow-step">
            <strong>4. Scanner checks hardware limits.</strong> Real systems verify
            RF power, SAR, gradient amplitude, slew rate, and timing. This simulator
            reports peak B1 and Gz, but does not model SAR or gradient slew limits.
        </div>
        <div class="workflow-step">
            <strong>5. The patient never sees the RF design.</strong> The scanner runs
            this hidden RF/gradient choreography repeatedly while collecting data
            that becomes the MRI image.
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_profile_detail(results: dict[str, np.ndarray | float], params: SimulationParams) -> None:
    """Render focused numbers and tables for the spatial excitation profile."""

    summary = calculate_design_summary(results, params)
    z_mm = np.asarray(results["z"]) * 1e3
    flip_angle = np.asarray(results["flip_angle"])
    final_m = np.asarray(results["final_m"])
    transverse = np.hypot(final_m[:, 0], final_m[:, 1])

    detail = pd.DataFrame(
        {
            "z_mm": z_mm,
            "flip_angle_deg": flip_angle,
            "transverse_magnitude": transverse,
            "Mz": final_m[:, 2],
        }
    )

    cols = st.columns([1.1, 1])
    with cols[0]:
        st.subheader("Slice profile samples")
        st.dataframe(
            detail.iloc[:: max(1, len(detail) // 32)].round(4),
            width="stretch",
            hide_index=True,
        )
    with cols[1]:
        st.subheader("Design readout with units")
        st.markdown(
            f"""
            - Target flip angle: `{params.flip_angle_deg:.0f} degrees`
            - Simulated center flip: `{summary['center_flip']:.2f} degrees`
            - Estimated slice thickness: `{summary['estimated_slice_mm']:.2f} mm`
            - RF bandwidth: `{summary['bandwidth_hz'] / 1e3:.2f} kHz`
            - Peak B1 amplitude: `{summary['peak_b1_ut']:.2f} uT`
            - Stopband maximum flip: `{summary['stopband_flip']:.2f} degrees`
            """
        )

    st.subheader("Profile trend")
    st.line_chart(
        detail.set_index("z_mm")[["flip_angle_deg", "transverse_magnitude", "Mz"]]
    )


def render_explanation() -> None:
    with st.expander("What the plots mean", expanded=True):
        st.markdown(
            """
            **Top left: RF pulse amplitude, B1(t), in microtesla (uT).**  
            This is the transverse magnetic field applied by the RF coil. The
            sinc shape creates a controlled frequency response. The Hamming
            window smooths the pulse to reduce ringing in the slice profile.

            **Top right: RF pulse spectrum, in kilohertz (kHz).**  
            This shows which resonance frequencies the pulse excites. A broader
            spectrum excites a broader range of positions when the gradient is on.

            **Bottom left: final magnetization, unitless.**  
            Magnetization starts at `[Mx, My, Mz] = [0, 0, 1]`. After the pulse,
            spins inside the slice should have transverse components `Mx`/`My`,
            while spins outside the slice should remain mostly at `Mz = +1`.

            **Bottom right: flip angle, in degrees.**  
            This is the easiest slice-profile view. Around the selected slice,
            the curve should approach the requested flip angle. Away from the
            slice, it should fall toward 0 degrees.

            The slice-selection gradient makes resonance frequency depend on
            position:

            `delta_omega(z) = gamma * Gz * z`

            `delta_omega` has units of rad/s, `gamma` has units of rad/s/T,
            `Gz` has units of T/m, and `z` has units of m. That relationship is
            the reason an RF frequency band becomes a spatial slab.

            The shaded region marks the requested slice thickness. The flip
            angle panel is the easiest way to read the slice profile.
            """
        )

    with st.expander("How the controls affect the result", expanded=True):
        st.markdown(
            """
            - Increasing **flip angle (degrees)** raises the intended rotation at
              the center of the slice. A 90 degree pulse creates transverse MRI
              signal; a 180 degree pulse inverts longitudinal magnetization.
            - Increasing **RF duration (ms)** generally lowers peak B1 for the
              same flip angle, but changes the bandwidth relationship.
            - Increasing **Gz (mT/m)** makes position map more strongly to
              frequency, so the same RF bandwidth selects a thinner slice.
            - Increasing **target slice thickness (mm)** asks the RF pulse to
              excite a wider slab of positions.
            - Increasing **TBW (unitless)** usually sharpens the slice edges, but
              can introduce more structure in the RF pulse and profile.
            """
        )


def main() -> None:
    inject_style()
    render_header()

    params = build_params()

    with st.spinner("Solving Bloch equations across spatial positions..."):
        results = run_from_params(params)

    render_quick_guide(params)
    render_metrics(results, params)

    console_tab, preset_tab, mri_tab, profile_tab, notes_tab = st.tabs(
        ["Pulse Console", "Slice Presets", "MRI Workflow", "Profile Detail", "Physics Notes"]
    )

    with console_tab:
        figure = make_main_figure(results, params)
        if figure is None:
            render_fallback_charts(results, params)
        else:
            st.plotly_chart(figure, use_container_width=True)

    with preset_tab:
        render_slice_preset_comparison(params)

    with mri_tab:
        render_mri_context(params, results)

    with profile_tab:
        render_profile_detail(results, params)

    with notes_tab:
        render_explanation()


if __name__ == "__main__":
    main()
