# 🌿 PlantLeaf – *Let plants speak*

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()
[![Award](https://img.shields.io/badge/🏆%20I%20Giovani%20e%20le%20Scienze-1st%20Place%202026-gold.svg)]()

> *Plants respond to stress with both electrical and acoustic signals. PlantLeaf gives you the tools to detect, visualise, and analyse them — with low-cost hardware and rigorous, open-source software.*

**[🌐 Website](https://plantleaf.it) · [📄 Documentation Repo](https://github.com/TommyVaninetti/PlantLeaf---documentation) · [📬 Contact](mailto:tommasovaninetti8@gmail.com)**

---

## Overview

PlantLeaf is a complete, open-source acquisition and analysis platform for plant ultrasonic and bioelectrical signals. It captures **ultrasonic click emissions** and **action-potential-like voltage responses** from stressed plants using custom low-cost hardware, and processes them in both is real time and offline through a cross-platform desktop application.

The project bridges plant biology, acoustic signal processing, and embedded systems — making advanced plant monitoring accessible without expensive commercial equipment.

---

## Screenshots

![iFFT reconstruction with Hilbert envelope](https://plantleaf.it/static/images/aloemate.png)

![Action Potential from Aloe light stress fitted](https://plantleaf.it/static/images/APM.png)

---

## Key Features

### 🔊 Ultrasonic Click Detection
- Real-time FFT visualisation of the 20–80 kHz ultrasonic band
- Automatic 4-stage click detection algorithm (v4.0) with sub-millisecond temporal resolution
- Time-domain signal reconstruction via inverse FFT (iFFT) with Gibbs suppression
- Hilbert envelope analysis for click morphology characterisation (decay constant τ, R²)
- Threshold-based and fully automatic detection modes

### 🔬 Voltage Signal Analysis
- Real-time acquisition and visualisation of plant electrical signals up to 1k samples/s
- Automatic mathematical fitting of action potential waveforms (sinusoidal depolarisation + exponential repolarisation)
- Energy calculation and correlation coefficient reporting for each detected event
- Export of raw voltage recordings and fitted parameters

### 🖥️ Application
- Cross-platform GUI (Windows, macOS, Linux) built with PySide6
- Unified interface for both voltage and audio acquisition and analysis
- Interactive spectrograms, time-averaged FFT graphs, and waveform inspection
- Full session export with per-click feature vectors (τ, R², SPR, amplitude, spectral ratio)

Complete documentation is available at here: [📄 Documentation Repo](https://github.com/TommyVaninetti/PlantLeaf---documentation)
---

## Hardware Requirements

PlantLeaf is designed around affordable, accessible components:

| Component | Description |
|-----------|-------------|
| **STM32F411CEU6** | Main microcontroller for both voltage and audio acquisition |
| **Knowles SPU0410LR5H-QB** | MEMS ultrasonic microphone (20–80 kHz) |
| **PlantLeaf Audio PCB** | Custom PCB for microphone, amplifier and filters |
| **PlantLeaf ESEB v1.0** | Custom PCB for electrical signal acquisition |

---

## Installation

```bash
pip install -r requirements.txt
python src/main.py
```

> Tested on Windows 11 and macOS 14.

---

## Detection Algorithm

The ultrasonic click detector is a 4-stage pipeline that processes continuous FFT streams from the STM32 firmware and identifies cavitation click candidates with high sensitivity and low false-positive rate:

| Stage | Operation |
|-------|-----------|
| **Stage 1** | Energy threshold (`μ + 5σ`) + run-length filter (sustained noise rejection) |
| **Stage 2** | Normalised peak FFT amplitude + Spectral Peak Ratio (broadband shape check) |
| **Stage 3** | Six-criterion temporal validation: amplitude, pre-click silence, energy decay, asymmetry, decay constant τ, exponential fit quality R² |
| **Stage 4** | Deduplication across consecutive frames |

📄 **Full technical specification:** [CLICK_DETECTION_ALGORITHM.md](https://github.com/TommyVaninetti/PlantLeaf---documentation/blob/main/App/Automatic_click_detection_algorithm/CLICK_DETECTION_ALGORITHM.md)

📄 **FFT and phase data specification:** [FFT_PHASE_TECHNICAL_SPECIFICATION.md](https://github.com/TommyVaninetti/PlantLeaf---documentation/blob/main/App/FFT_and_acquisition_specifications/FFT_PHASE_TECHNICAL_SPECIFICATION.md)

---

## Experimental Results

Our research is fully available on [plantleaf.it](https://plantleaf.it), includind raw recordings, spectrograms, and annotated click datasets, all available for download in our database.
We have led experiments on Aloe Vera, Ferrocactus and Dionea. We are looking to strengthen the results we have already obtained but also to try our system with other plants and other environments.

---

## Future Developments

We are actively developing our software and hardware, in particular:
- Ultrasonic Clicks detector algorithm v5: adaptive threshold, SVM algorithm are to be introduced for analysis in non-noisy environments as well
- ASEB and wireless instrumentation: currently testing the ASEB and developing a wireless module with automatic click detection
- Physical Simulators: made to simulate both ultrasonic clicks and action potentials on computer

## Project Structure

```
PlantLeaf-Desktop-App/
│
├── src/
│   ├── main.py                          # Application entry point
│   │
│   ├── components/                      # Reusable UI widgets
│   │   ├── batch_click_csv_export.py
│   │   ├── batch_export_screenshots.py
│   │   ├── choose_serial_port.py
│   │   ├── click_detector_dialog.py
│   │   ├── data_table.py
│   │   ├── multi_file_batch_export.py
│   │   ├── not_saved_popup.py
│   │   ├── sampling_settings.py
│   │   ├── start_stop_button.py
│   │   ├── time_input_widget.py
│   │   └── trim_region_dialog.py
│   │
│   ├── config/
│   │   └── app_config.py                # Application-wide constants
│   │
│   ├── core/                            # Base classes and managers
│   │   ├── base_window.py
│   │   ├── replay_base_window.py        # Math/analysis dialog (MathOperations)
│   │   ├── file_handler_mixin.py
│   │   ├── audio_trim_export.py
│   │   ├── voltage_trim_export.py
│   │   ├── font_manager.py
│   │   ├── theme_manager.py
│   │   ├── layout_manager.py
│   │   ├── settings_manager.py
│   │   ├── special_component.py
│   │   └── wake_lock_manager.py         
│   │
│   ├── plotting/
│   │   └── plot_manager.py              # pyqtgraph wrappers
│   │
│   ├── saving/
│   │   ├── audio_save_worker.py         # Async .paudio file writer
│   │   ├── audio_load_progress.py
│   │   └── voltage_save_worker.py       # Async .pvolt file writer
│   │
│   ├── serial_communication/
│   │   ├── audio_reader.py              # STM32 USB CDC audio stream parser
│   │   └── voltage_read.py              # STM32 USB CDC voltage stream parser
│   │
│   └── windows/
│       ├── main_window_home.py          # Home / experiment selector
│       ├── main_window_audio.py         # Real-time audio acquisition window
│       ├── main_window_voltage.py       # Real-time voltage acquisition window
│       ├── replay_window_audio.py       # Offline audio replay & analysis
│       ├── replay_window_voltage.py     # Offline voltage replay & analysis
│       └── ui/                          # PySide6 UI files created with QtCreator
│           ├── ui_MainWindowAudio.py
│           ├── ui_MainWindowVoltage.py
│           └── ui_MathDialog.py
│
├── assets/
│   ├── logo.png / logo.ico / logo_for_app.icns
│   └── icons/                           # Toolbar action icons (PNG)
│
├── themes/                              # QSS stylesheets
│   ├── dark.css / light.css
│   └── dark_amber/blue/green  ·  light_amber/blue/green
│
├── csv_to_pvolt.py                      # Utility: convert CSV to .pvolt format
├── licenses.txt                         # Third-party licence notices
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE                              # AGPL-3.0
└── README.md
```

---

## Dependencies

| Library | Version | License |
|---------|---------|---------|
| PySide6 | 6.9.0 | LGPLv3 |
| pyqtgraph | latest | MIT |
| pyserial | latest | BSD |
| numpy | latest | BSD |
| scipy | latest | BSD |
| wakepy | latest | MIT |

---

## Contributing

Bug reports, feature suggestions, and pull requests are welcome.
Please read [CONTRIBUTING.md](CONTRIBUTING.md) before opening an issue or PR.

For questions about the detection algorithm or experimental methodology, feel free to open a Discussion or reach out via the website.

---

## Citation

If you use PlantLeaf in your research or build upon it, please cite it as:

```bibtex
@software{vaninetti2026plantleaf,
  author       = {Vaninetti, Tommaso},
  title        = {{PlantLeaf}: An Open-Source Platform for Plant Ultrasonic and Bioelectrical Signal Acquisition and Analysis},
  year         = {2026},
  license      = {AGPL-3.0},
  url          = {https://github.com/TommyVaninetti/PlantLeaf-Desktop-App},
  note         = {overall 1st place, I Giovani e le Scienze 2026 (Italy). Competing at EUCYS 2026.}
}
```

> If you reference the detection algorithm specifically, please also cite the accompanying technical documentation linked above.

---

## License

Copyright (C) 2026 Tommaso Vaninetti.
Licensed under the **GNU Affero General Public License v3.0** — see [LICENSE](LICENSE) for details.
