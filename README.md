# MRI Slice-Selective RF Pulse Designer

> Public archive note: this repository is a portfolio/demo-safe version prepared from private working repositories/materials; sensitive details, credentials, raw logs, and proprietary context are intentionally omitted.

Interactive Python simulator for MRI slice-selective RF excitation.

The project connects the MRI operator's view of a scan prescription with the
hidden pulse-design layer underneath it. An operator thinks in terms of slice
thickness, flip angle, scan plane, and sequence type. The scanner turns those
choices into RF pulse amplitude, RF bandwidth, slice-selection gradient, and
timing. This app visualizes that conversion with a Bloch-equation simulation.

## What It Simulates

- Hamming-windowed sinc RF pulse
- Constant slice-selection gradient `Gz`
- Position-dependent off-resonance, `delta_omega(z) = gamma * Gz * z`
- Rotating-frame Bloch dynamics without relaxation
- Final magnetization and flip angle across spatial position

## Run Locally

```powershell
pip install -r requirements.txt
streamlit run app.py
```

The app opens a browser dashboard with controls for:

- flip angle, in degrees
- RF pulse duration, in ms
- slice-selection gradient, in mT/m
- target slice thickness, in mm
- time-bandwidth product
- numerical resolution

It also includes scanner-style presets and a comparison tab for overlaying
multiple Bloch-equation slice profiles, including thin-slice, localizer,
sharp-edge, low-gradient, and inversion pulse scenarios.

## Files

- `app.py`: Streamlit dashboard
- `simulation.py`: standalone Bloch-equation simulator
- `project.html`: website-ready project page
- `project.json`: compact project metadata for a portfolio data file
- `assets/simulation-preview.png`: simulation preview plot

## Website Integration

This repository is meant to be linked from a portfolio project entry. The
`project.json` file contains the copy and metadata needed for a website project
section, while `project.html` is a standalone page that can be adapted into the
site's existing design system.
