# FootAnalytic — Foot Weight Distribution & Plantar Pressure Analytics (32×32 FSR)

**FootAnalytic** is an open research project for measuring and analyzing **plantar pressure / foot weight distribution** using a **32×32 Force-Sensitive Resistor (FSR) sensor array** (1,024 sensing elements).  
It provides a full pipeline: **embedded acquisition → calibration → feature extraction → standardized reference ranges → ML/DL classification → visualization (2D heatmaps + 3D plots) → cloud reporting**.

> **Primary use-case:** objective screening and monitoring of musculoskeletal loading patterns during **quiet standing** (static stance).  
> **Note:** This study does **not** include dynamic gait trials.

---

## Highlights (from the manuscript)

### ✅ Large cohort with fitness labeling
- **4,000 adults** completed a Physical Fitness Assessment (PFA).
- **2,200** classified as **musculoskeletally fit** (used to build reference ranges).
- **1,800** classified as **unfit** (used for comparison and classification).

### ✅ Standardized reference ranges (quiet standing)
Musculoskeletally fit adults typically load:
- **Heel:** ~45–55%
- **Midfoot:** ~10–15%
- **Metatarsals:** ~17–27%
- **Toes:** ~8–13%

### ✅ Validated low-cost hardware
- **32×32 FSR matrix (Rx-M3232L)** scanned via **ADG731 multiplexer**
- Signal conditioning: **Wheatstone bridge + TI INA333 instrumentation amplifier**
- Embedded acquisition: **Canaan K230 (RISC-V)** with **12-bit ADC @ 100 Hz**
- Reliability: **test–retest < 3%** (regional load)
- Force-plate validation (10 participants): **total force within ~5%** of lab force plate, **regional correlation r > 0.95**

### ✅ Machine learning / deep learning baselines
- **SVM (Gaussian kernel)** using regional %, ratios, descriptive stats + PCA scores: **~90% accuracy**
- **CNN** trained on full 32×32 pressure maps (80/20 train/validation): **~95% accuracy**
- CNN architecture (as reported): 2 conv layers (5×5) + max-pooling + 2 FC layers + softmax (fit vs unfit)

---

## Live demo (cloud analytics)
A hosted instance (if available to you) can be accessed here:

- **http://cloud.pakhpc.com:8505/**

> If the demo is restricted, contact the authors/maintainers for access.

---

## Dataset & File Format

The repository includes **de-identified** plantar pressure maps and associated attributes used in the analyses.

**Pressure maps**
- Each sample is a **32×32 matrix** exported as **CSV**
- Stored as integer values (derived from the embedded acquisition pipeline)

**Metadata / attributes (examples from the paper)**
- Subject ID (pseudonymous)
- Age, height, weight
- Foot length/width
- “Dysfunctions” / physical issues label (if present in your dataset schema)
- Fitness label: **fit vs unfit** (from PFA)

---

## Installation

### 1) Clone and install dependencies
```bash
git clone https://github.com/ucerd/footanalytic.git
cd footanalytic

python -m venv .venv
# Linux/Mac:
source .venv/bin/activate
# Windows:
# .venv\Scripts\activate

pip install -r requirements.txt
