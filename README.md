# PlantLeaf – *Let plants speak*

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Version](https://img.shields.io/badge/version-1.0.0-green.svg)]()
[![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey.svg)]()
[![Award](https://img.shields.io/badge/🏆%20I%20Giovani%20e%20le%20Scienze-1st%20Place%202026-gold.svg)]()

> *Plants respond to stress with both electrical and acoustic signals. PlantLeaf gives you the tools to detect, visualise, and analyse them — with low-cost hardware and rigorous, open-source software.*

**[Website](https://plantleaf.it) · [Documentation Repo](https://github.com/TommyVaninetti/PlantLeaf---documentation) · [Contact](mailto:tommasovaninetti8@gmail.com)**

---

## Overview

PlantLeaf is a complete, open-source acquisition and analysis platform for plant ultrasonic and bioelectrical signals. It captures **ultrasonic click emissions** and **action-potential-like voltage responses** from stressed plants using custom low-cost hardware, and processes them in both is real time and offline through a cross-platform desktop application.

The project bridges plant biology, acoustic signal processing, and embedded systems — making advanced plant monitoring accessible without expensive commercial equipment.

---

## Key Features

### Ultrasonic Click Detection
- Real-time FFT visualisation of the 20–80 kHz ultrasonic band at 390 FPS
- 4-stage click detection pipeline v5: adaptive noise floor, hard spectral gates, SVM classifier, deduplication
- Time-domain signal reconstruction via inverse FFT (iFFT) with Gibbs suppression and Hilbert envelope
- 17 acoustic features per candidate: SNR ratios, decay constant τ, R², ZCR, kurtosis, spectral shape
- Interactive Stage 1 threshold filter with adaptive noise floor display during replay
- Batch data collection export: CSV of 17 features + screenshots for manual labelling and model training

### Machine Learning Pipeline
- SVM classifier (RBF kernel, scikit-learn Pipeline) trained on 285 labeled candidates across 4 species
- Session-level cross-validation (`StratifiedGroupKFold`) to prevent recording-level data leakage
- AUC-ROC = 0.835; Set B recall = 0.962 at threshold 0.220 (optimised for recall ≥ 0.90)
- Offline tools: `train_svm.py`, `evaluate_candidates.py`, `analyze_dataset.py` with per-file confusion matrices and click-rate plots

### Voltage Signal Analysis
- Real-time acquisition and visualisation of plant electrical signals up to 1 kHz
- Automatic mathematical fitting of action potential waveforms (sinusoidal depolarisation + exponential repolarisation)
- R² goodness-of-fit and signal energy reported per event
- Export of raw voltage recordings and fitted parameters

### Application
- Cross-platform GUI (Windows, macOS, Linux) built with PySide6
- Unified interface for both voltage and audio acquisition and analysis
- Interactive spectrograms, time-averaged FFT energy plots, iFFT waveform inspection
- Full session export with per-click feature vectors

Complete documentation is available here: [Documentation Repo](https://github.com/TommyVaninetti/PlantLeaf---documentation)

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

The current algorithm (v5) is a 4-stage pipeline that processes continuous FFT streams from the STM32 firmware and identifies cavitation click candidates with high recall:

| Stage | Operation |
|-------|-----------|
| **Stage 1** | Adaptive energy threshold: `E_i > k × Ê_floor` — per-frame noise floor estimated by `AdaptiveNoiseEstimatorV5` |
| **Stage 2** | Hard gates: R² ≥ 0.10 (exponential decay quality) and SPR < 100 (broadband shape) |
| **Stage 3** | SVM classifier (`SimpleImputer → StandardScaler → SVC`, RBF kernel, C=50, γ=0.01) on 16 acoustic features; threshold = 0.220 |
| **Stage 4** | Deduplication across consecutive frames |

**v5 full specification (current):** [CLICK_DETECTION_ALGORITHM_v5.md](https://github.com/TommyVaninetti/PlantLeaf---documentation/blob/main/App/Automatic_click_detection_algorithm/CLICK_DETECTION_ALGORITHM_v5.md)

**v4 specification (historical):** [CLICK_DETECTION_ALGORITHM_v4.md](https://github.com/TommyVaninetti/PlantLeaf---documentation/blob/main/App/Automatic_click_detection_algorithm/CLICK_DETECTION_ALGORITHM_v4.md)

**FFT and phase data specification:** [FFT_PHASE_TECHNICAL_SPECIFICATION.md](https://github.com/TommyVaninetti/PlantLeaf---documentation/blob/main/App/FFT_PHASE_TECHNICAL_SPECIFICATION.md)

## Experimental Results

Our research is fully available on [plantleaf.it](https://plantleaf.it), includind raw recordings, spectrograms, and annotated click datasets, all available for download in our database.
We have led experiments on Aloe Vera, Ferrocactus and Dionea. We are looking to strengthen the results we have already obtained but also to try our system with other plants and other environments.

---

## Future Developments

We are actively developing our software and hardware, in particular:
- ASEB and wireless instrumentation: currently testing the ASEB board and developing a wireless module with on-device click detection
- Expanding the SVM training dataset across more plant species and stress conditions
- Physical simulators: tools to simulate both ultrasonic clicks and action potentials for algorithm validation

## Project Structure

```
PlantLeaf-Desktop-App/
│
├── src/
│   ├── main.py                          # Application entry point
│   │
│   ├── components/                      # Reusable UI widgets
│   │   ├── data_collection_dialog_v5.py # Batch Stage-1 export: CSV + screenshots
│   │   ├── choose_serial_port.py
│   │   ├── data_table.py
│   │   ├── not_saved_popup.py
│   │   ├── sampling_settings.py
│   │   ├── start_stop_button.py
│   │   ├── time_input_widget.py
│   │   └── trim_region_dialog.py
│   │
│   ├── config/
│   │   └── app_config.py                # Application-wide constants
│   │
│   ├── core/                            # Base classes and pipeline
│   │   ├── click_pipeline_v5.py         # Full v5 detection pipeline + 17 features
│   │   ├── replay_base_window.py        # Shared replay UI (MathOperations dialog)
│   │   ├── base_window.py
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
│   ├── ml/                              # Offline machine learning scripts
│   │   ├── train_svm.py                 # SVM training with session-level CV
│   │   ├── evaluate_candidates.py       # Batch inference on candidate CSVs
│   │   └── analyze_dataset.py           # Per-file stats, confusion matrix, plots
│   │
│   ├── plotting/
│   │   └── plot_manager.py              # PyQtGraph wrappers
│   │
│   ├── saving/
│   │   ├── audio_save_worker.py         # Async .paudio file writer
│   │   ├── audio_load_progress.py
│   │   └── voltage_save_worker.py       # Async .pvoltage file writer
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
│       └── ui/                          # PySide6 UI files (Qt Designer)
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
│   └── dark_amber / dark_blue / dark_green  ·  light_amber / light_blue / light_green
│
├── csv_to_pvolt.py                      # Utility: convert CSV to .pvoltage format
├── licenses.txt                         # Third-party licence notices
├── requirements.txt
├── CONTRIBUTING.md
├── LICENSE                              # AGPL-3.0
└── README.md
```

---

## Screenshots

![iFFT reconstruction with Hilbert envelope](https://plantleaf.it/static/images/aloemate.png)

![Action Potential from Aloe light stress fitted](https://plantleaf.it/static/images/APM.png)

---

## Dependencies

| Library | Version | License | Used in |
|---------|---------|---------|---------|
| PySide6 | 6.9.0 | LGPLv3 | App |
| PyQtGraph | latest | MIT | App |
| NumPy | latest | BSD | App + ML scripts |
| SciPy | latest | BSD | App + ML scripts |
| PySerial | latest | BSD | App |
| wakepy | latest | MIT | App |
| scikit-learn | 1.6.1 | BSD | App (Stage 3 inference) + ML scripts |
| joblib | 1.5.3 | BSD | App (model loading) + ML scripts |
| pandas | 2.3.3 | BSD | ML scripts only |
| matplotlib | 3.9.4 | BSD | ML scripts only |

See [LIBRARIES.md](https://github.com/TommyVaninetti/PlantLeaf---documentation/blob/main/App/LIBRARIES.md) for the full rationale behind each choice.

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
