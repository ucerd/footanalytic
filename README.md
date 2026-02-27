# FootAnalytic: Foot Pressure Analysis System

FootAnalytic is an open research project for measuring and analyzing human foot weight distribution using a 32×32 FSR sensor array.  We collected a large dataset of plantar pressure maps (N=2200 healthy adults) and developed machine-learning models (SVM, CNN) to detect abnormal pressure patterns. The codebase includes data processing scripts, ML models, and real-time analysis software. The associated manuscript details the methods and results.

## Features
- Standardized foot weight distribution patterns for healthy subjects.
- Raspberry Pi / RISC-V based embedded system with 32×32 sensor grid.
- Real-time data acquisition and cloud reporting (Python scripts in `src/`).
- SVM and CNN models for classifying weight distribution (Jupyter examples in `examples/`).
- Full dataset (anonymized CSVs) released under [CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/).

## Installation
```bash
git clone https://github.com/ucerd/footanalytic.git
cd footanalytic
pip install -r requirements.txt

