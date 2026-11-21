# 🌧️ MahaMonsoon-X: Advanced Progressive Spatiotemporal Rainfall Intelligence
### *Comparative Spatiotemporal Modeling & Benchmarking for Regional Monsoon Rainfall over Maharashtra*

---

[![Python Version](https://img.shields.io/badge/Python-3.12.10%20Required-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/downloads/release/python-31210/)
[![Environment Setup](https://img.shields.io/badge/Setup-pyproject.toml%20%7C%20Automated%20Scripts-success.svg)](#-environment-setup--execution-guide)
[![Framework](https://img.shields.io/badge/Framework-PyTorch%20%7C%20Scikit--Learn-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Dataset](https://img.shields.io/badge/Dataset-ECMWF%20ERA5%20Reanalysis-4A90E2.svg)](https://cds.climate.copernicus.eu/)
[![Study Region](https://img.shields.io/badge/Region-Maharashtra%20%2814.75°--22.75°N%2C%2065.25°--82.25°E%29-orange.svg)](#-geospatial-domain--grid-architecture)
[![Evaluation Focus](https://img.shields.io/badge/Scope-Benchmark%20%26%20Model%20Comparison-success.svg)](#-model-comparisons--evaluation-results)

---

## 📌 Executive Summary

**MahaMonsoon-X** is an empirical, research-grade meteorological evaluation framework designed to benchmark *( comparision )* spatiotemporal deep learning models against traditional statistical and supervised baselines for regional monsoon rainfall forecasting. Focus is directed onto the **Indian summer monsoon season (JJAS: June, July, August, September)** across Maharashtra and surrounding peninsular/coastal domains.

The central research inquiry tests **whether the steep computational complexity and structural overhead of spatiotemporal recurrent networks (ConvLSTM) yield genuine, measurable predictive advantages over lightweight historical climatological moving averages (AIMRI) and multivariate regularized regression (Multiple Linear Regression)**.

> ⚠️ **Scientific Boundary & Limitation:**  
> This evaluation suite exclusively utilizes **ECMWF ERA5 monthly reanalysis data (1940–2026)**. Target precipitation ($tp$) represents ERA5 monthly total precipitation rather than ground-based rain-gauge observations (IMD). Metric outputs evaluate the predictive capability over ERA5 meteorological precipitation fields and should not be confused with uncorrected in-situ gauge measurements.

---

## 🏛️ System Architecture & Workflow Pipeline

```text
========================================================================================================================
                                      MAHAMONSOON-X END-TO-END WORKFLOW PIPELINE
========================================================================================================================

  +------------------------------------------------------------------------------------------------------------------+
  |                                        STAGE 1: DATA INGESTION & AUDITING                                        |
  |  • ECMWF Copernicus ERA5 Reanalysis (1940-2026 Monthly Single-Levels GRIB)                                      |
  |  • Notebook: 01_era5_data_audit.ipynb (Headers scan, duplicate timestamps check, spatial bounding coordinates)   |
  +------------------------------------------------------------------------------------------------------------------+
                                                           |
                                                           v
  +------------------------------------------------------------------------------------------------------------------+
  |                                     STAGE 2: PREPROCESSING & HARMONIZATION                                       |
  |  • Notebook: 02_preprocess_era5.ipynb                                                                            |
  |  • Hypercube Flattening  --> Regular coordinate table: (timestamp, latitude, longitude)                          |
  |  • Quality Assurance     --> Dropped variables with >50% missing values; median-imputed small gaps               |
  |  • Export Target         --> data/processed/era5_processed.csv [15 meteorological predictors + 'tp' target]      |
  +------------------------------------------------------------------------------------------------------------------+
                                                           |
               +-------------------------------------------+------------------------------------------+
               |                                           |                                          |
               v                                           v                                          v
  +-------------------------+                 +-------------------------+                +-------------------------+
  |   MODEL 1: STATISTICAL  |                 |    MODEL 2: REGRESSION  |                |   MODEL 3: DEEP LEARNING|
  |                         |                 |                         |                |                         |
  |   03_baseline_aimri     |                 |   04_multiple_linear_   |                |   05_convlstm_          |
  |         .ipynb          |                 |     regression.ipynb    |                |     forecasting.ipynb   |
  |                         |                 |                         |                |                         |
  |  • AIMRI Climatological |                 |  • Regularized Ridge    |                |  • Spatiotemporal       |
  |    Moving Average       |                 |    Regression           |                |    ConvLSTM Network     |
  |  • 5-Year Same-Calendar |                 |  • May Pre-Monsoon      |                |  • 5-Month Rolling      |
  |    Month Lookback (K=5) |                 |    Atmosphere + Lags    |                |    (B, T, C, H, W) Cube |
  |  • Non-parametric,      |                 |  • Tabular Multivariate |                |  • Recurrent Spatial    |
  |    zero parameter drift |                 |    Mapping              |                |    Convolutions (3x3)   |
  +-------------------------+                 +-------------------------+                +-------------------------+
               |                                           |                                          |
               +-------------------------------------------+------------------------------------------+
                                                           |
                                                           v
  +------------------------------------------------------------------------------------------------------------------+
  |                                   STAGE 4: COMMON-SUPPORT BENCHMARK & RANKING                                    |
  |  • Notebook: 06_model_comparison.ipynb                                                                           |
  |  • Rigorous Intersection --> 50 shared sampled test years (1975-2025) across 910,800 identical grid points       |
  |  • Evaluation Metrics    --> Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), Residual Distribution    |
  |  • Output Artifacts      --> model_comparison_metrics.csv, model_ranking.csv, comparison_report.json            |
  +------------------------------------------------------------------------------------------------------------------+
========================================================================================================================
```

### 🔄 Pipeline Stage Breakdown

| Stage | Primary Notebook | Key Operations & Transformation Logic | Artifacts Generated |
|:---:|---|---|---|
| **1. Data Ingestion** | `01_era5_data_audit.ipynb` | Metadata scan, dimension extraction, coordinate sorting, duplicate checks. | `src/artifacts/audit/` |
| **2. Preprocessing** | `02_preprocess_era5.ipynb` | Flattening GRIB hypercubes, filtering sparse columns ($>50\%$ missing), training-safe median imputation. | `data/processed/era5_processed.csv` |
| **3. Baseline AIMRI** | `03_baseline_aimri.ipynb` | Deterministic $5$-year historical same-calendar-month spatial moving average ($K=5$). | `src/artifacts/aimri/` |
| **4. Linear Regression** | `04_multiple_linear_regression.ipynb` | Multivariate Ridge regression using autoregressive rainfall lags ($1\text{–}5$) and May synoptic conditions. | `src/artifacts/linear_regression/` |
| **5. ConvLSTM Network** | `05_convlstm_forecasting.ipynb` | Spatiotemporal tensor modeling $(B, T, C, H, W)$, $3\times3$ spatial recurrent convolutions, early stopping. | `src/artifacts/convlstm/` |
| **6. Model Comparison** | `06_model_comparison.ipynb` | Intersection across $50$ common test years ($910,800$ rows) to evaluate MAE and RMSE without distribution bias. | `src/artifacts/model_comparison/` |

---

## ⚙️ Mathematical Formulations of Compared Models

### 1. AIMRI (Adaptive Historical Climatological Mean)
AIMRI acts as a non-parametric, deterministic historical moving average. For year $y$, target calendar month $m \in \{6, 7, 8, 9\}$, and spatial coordinate $(i, j)$ with lookback window $K=5$:

$$\widehat{P}_{y, m, i, j} = \frac{1}{K} \sum_{k=1}^{K} P_{y-k, m, i, j}$$

- **Characteristics:** Zero gradient updates, zero trainable parameters, immune to optimization instability, perfectly preserves local seasonal climatology.

---

### 2. Multiple Linear Regression (Regularized Ridge)
Maps atmospheric pre-monsoon predictors (May conditions) along with autoregressive precipitation lags directly to downstream monsoon precipitation:

$$\widehat{y}_{t, g} = \mathbf{w}^T \mathbf{x}_{t, g} + b, \quad \min_{\mathbf{w}, b} \left\{ \sum_{t, g} (y_{t, g} - \mathbf{w}^T \mathbf{x}_{t, g} - b)^2 + \alpha \|\mathbf{w}\|_2^2 \right\}$$

- **Predictor Schema:** Autoregressive lags $\text{rain\_lag\_1}$ through $\text{rain\_lag\_5}$ combined with 15 pre-monsoon (May) atmospheric indicators.

---

### 3. Spatiotemporal ConvLSTM (Convolutional LSTM)
Replaces the standard dense matrix multiplications in LSTM cell gates with spatial 2D convolution operations ($*$), retaining continuous geographic adjacency:

$$\begin{aligned}
i_t &= \sigma(W_{xi} * \mathcal{X}_t + W_{hi} * \mathcal{H}_{t-1} + W_{ci} \odot \mathcal{C}_{t-1} + b_i) \\
f_t &= \sigma(W_{xf} * \mathcal{X}_t + W_{hf} * \mathcal{H}_{t-1} + W_{cf} \odot \mathcal{C}_{t-1} + b_f) \\
\mathcal{C}_t &= f_t \odot \mathcal{C}_{t-1} + i_t \odot \tanh(W_{xc} * \mathcal{X}_t + W_{hc} * \mathcal{H}_{t-1} + b_c) \\
o_t &= \sigma(W_{xo} * \mathcal{X}_t + W_{ho} * \mathcal{H}_{t-1} + W_{co} \odot \mathcal{C}_t + b_o) \\
\mathcal{H}_t &= o_t \odot \tanh(\mathcal{C}_t)
\end{aligned}$$

- **Input Tensor Dimensionality:** $(B, T, C, H, W)$ where $T=5$ lookback months, $C=32$ channels (15 meteorological variables + missingness masks + historical target channels), $H=33$ latitude bins, $W=69$ longitude bins.

---

## 📊 Comprehensive Input & Target Parameters

| Feature Variable | ERA5 Technical Code | Atmospheric / Surface Role | Physical Meaning |
|---|---|---|---|
| **Total Precipitation (Target)** | `tp` | **Target Predictand ($y$)** | Monthly total water equivalent precipitation |
| **Zonal 10m Wind** | `10u` | Kinematic Circulation | East-West wind velocity vector at 10m height |
| **Meridional 10m Wind** | `10v` | Kinematic Circulation | North-South monsoon surge vector at 10m height |
| **Neutral Wind Speed** | `10si` | Wind Dynamics | Scalar magnitude of near-surface wind speed |
| **2m Temperature** | `2t` | Thermodynamics | Surface air thermal energy content |
| **2m Dewpoint Temperature** | `2d` | Moisture Saturation | Near-surface humidity and water-holding boundary |
| **Skin Temperature** | `skt` | Surface Interface | Radiative equilibrium temperature of ground/sea |
| **Surface Pressure** | `sp` | Dynamic Topography | Local barometric weight at terrain level |
| **Mean Sea Level Pressure** | `msl` | Synoptic Meteorology | Normalized barometric pressure (Monsoon Trough) |
| **Total Column Water Vapour** | `tcwv` | Moisture Reservoir | Precipitable water throughout the entire atmospheric column |
| **Vertically Integrated Moisture Divergence** | `avg_vimdf` | Moisture Dynamics | Horizontal convergence/divergence fueling convective storm cells |
| **Surface Latent Heat Flux** | `slhf` | Thermodynamic Flux | Energy transferred via ocean/land water phase changes |
| **Surface Solar Radiation Downwards** | `ssrd` | Radiative Forcing | Insolation warming the subcontinent |
| **Top Net Thermal Radiation** | `ttr` | Radiation Budget | Outgoing longwave thermal balance |
| **Low Vegetation Leaf Area Index** | `lai_lv` | Biosphere Coupling | Seasonal green canopy evapotranspiration capacity |
| **Volumetric Soil Water (Layer 1)** | `swvl1` | Terrestrial Memory | Near-surface root-zone soil saturation (0–7 cm depth) |

---

## 🗺️ Geospatial Domain & Grid Architecture

```
Latitude Span  :  14.75° N  to  22.75° N   (Δ = 0.25°, 33 coordinate steps)
Longitude Span :  65.25° E  to  82.25° E   (Δ = 0.25°, 69 coordinate steps)
Spatial Extent :  Maharashtra Heartlands, Western Ghats orographic barrier, 
                  Konkan Coast, and adjoining Arabian Sea & Bay of Bengal sectors
Total Grid Matrix: 33 × 69 = 2,277 spatial nodes per time step
```

---

## 📈 Model Comparisons & Evaluation Results

All models were evaluated across **50 common test years** (1975–2025) over identical year-month-grid intersections, totaling **910,800 evaluated space-time coordinate predictions**.

| Performance Rank | Predictive Model Architecture | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Primary Failure Modes & Limitations |
|:---:|---|:---:|:---:|---|
| 🥇 **1** | **AIMRI** *(Historical Climatology)* | **0.7267** | **0.9228** | Lacks dynamic sensitivity; cannot anticipate interannual anomalies or climate extremes |
| 🥈 **2** | **ConvLSTM** *(Spatiotemporal Recurrent)* | **0.7488** | **0.9171** | Slights spatial smoothing; requires significant compute; susceptible to vanishing gradients |
| 🥉 **3** | **Multiple Linear Regression** *(Ridge)* | **0.7739** | **1.4309** | Suffers severe variance on extreme values; ignores non-linear atmospheric feedback |

> 📌 **Key Metric Insight:** ConvLSTM achieves the lowest Root Mean Squared Error (**0.9171**), demonstrating strong mitigation of large outlying prediction errors due to spatial convolution filtering. However, AIMRI claims the lowest Mean Absolute Error (**0.7267**), proving that local seasonal climatology forms an exceptionally rigorous baseline that complex networks struggle to beat.

---

## ❓ Critical Research & Engineering Questions (FAQ)

### Q1: Why compare a complex network (ConvLSTM) against simple baselines like AIMRI and Linear Regression?
> **Answer:**  
> In applied environmental deep learning, complex neural architectures often show apparent success merely because simple, domain-specific baselines were omitted. We structured this tri-model benchmark to rigorously test **whether the high computational cost, GPU training overhead, and architectural hyperparameter tuning of deep spatiotemporal recurrent models (ConvLSTM) actually yield actionable, meaningful accuracy gains over lightweight statistical baselines**. Establishing AIMRI and Ridge Regression reveals the true value-add of non-linear spatial convolutions versus local historical memory.

---

### Q2: How did you preprocess and structure the ERA5 data so all three models could be fairly evaluated?
> **Answer:**  
> The workflow executed a standardized, five-stage transformation pipeline:
> 1. **Header & Integrity Auditing:** Scanned raw GRIB messages (`01_era5_data_audit.ipynb`) to confirm coordinates, variable definitions, and eliminate duplicate timestamp entries.
> 2. **Tabular Harmonization:** Converted multi-dimensional GRIB cubes into a unified coordinate table (`02_preprocess_era5.ipynb`) indexed by `(timestamp, latitude, longitude)`.
> 3. **Missing Value Filtering & Imputation:** Dropped columns with $>50\%$ missing values, and imputed remaining minor gaps using training-period medians.
> 4. **Model-Specific Structuring:**
>    - *Linear Regression:* Formulated flat matrices with autoregressive target lags ($\text{rain\_lag\_1}$ to $5$) and May indicators.
>    - *AIMRI:* Maintained index lookups for historical same-month target queries across a 5-year rolling window.
>    - *ConvLSTM:* Reconstructed a regular 5D tensor cube $(B, T, C, H, W)$ spanning $33 \times 69$ spatial coordinates across 32 input channels.
> 5. **Common Evaluation Intersect:** Intersected predictions on identical $(year, month, lat, lon)$ tuples (`06_model_comparison.ipynb`) across 50 shared years (910,800 rows).

---

### Q3: Where did Linear Regression and AIMRI fail compared to ConvLSTM?
> **Answer:**  
> - **Multiple Linear Regression Failures:** Lacks spatial awareness. Because it models rows independently without neighbor convolution, it cannot capture organized cloud clusters or orographic boundary dynamics along the Western Ghats. It produced an **RMSE of 1.4309** (56% higher error dispersion than ConvLSTM), suffering severe variance when rainfall spikes occurred.
> - **AIMRI Failures:** Purely static. AIMRI has zero awareness of contemporary atmospheric conditions, winds, sea surface temperature, or pressure depressions. During anomalous monsoon years (e.g., severe El Niño droughts or excess cyclonic flooding), AIMRI outputs historical averages, failing to capture regime shifts that ConvLSTM identifies through real-time moisture divergence and wind fields.

---

### Q4: Why is conventional accuracy percentage not reported for rainfall evaluation?
> **Answer:**  
> Rainfall prediction is a continuous real-valued spatial regression problem. In continuous regression, classification "accuracy" (e.g., $95\%$) does not exist mathematically. Using normalized percentages ($1 - \text{relative error}$) creates misleading impressions because precipitation variance across space and time spans orders of magnitude (from zero drizzle to extreme cloudburst events). We adhere strictly to standard meteorological verification metrics: **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.

---

### Q5: What is the significance of the May pre-monsoon information boundary?
> **Answer:**  
> To prevent **temporal data leakage**, forecasting models cannot consume future weather data. For a forecast issued for the monsoon season (June–September), the model is restricted to meteorological predictor fields observable up to May. This mirrors true operational early-warning forecasting where predictors from subsequent months are unavailable at inference time.

---

### Q6: Why did AIMRI achieve a lower MAE than ConvLSTM despite having no neural parameters?
> **Answer:**  
> Monsoon rainfall over peninsular India exhibits powerful local climatological regularity dictated by fixed topography (e.g., the orographic lifting of the Western Ghats vs. the rain-shadow Deccan plateau). AIMRI implicitly memorizes this permanent local spatial distribution through historical averaging. In contrast, neural networks optimize an empirical global loss function that often introduces slight spatial smoothing, resulting in minor baseline offsets across low-rainfall grid cells that incrementally inflate MAE.

---

## 📂 Project Directory Structure

```text
MahaMonsoon-X/
├── .gitignore                             # Git ignore rules for data, virtual environments, caches
├── .python-version                        # Declares Python 3.12.10 pinning (pyenv / rye / uv)
├── pyproject.toml                         # Standard PEP 517/518 build metadata & dependencies
├── setup_env.bat                          # Automated 1-click Windows setup script (Python 3.12.10)
├── setup_env.sh                           # Automated 1-click Unix/Linux/macOS/WSL setup script
├── LICENSE                                # MIT Open Source License
├── README.md                              # Project documentation and research report
├── requirements.txt                       # Locked dependencies for Python 3.12.10
├── data/
│   ├── era5/
│   │   ├── download_dataset.py            # Automated Copernicus CDS extraction script
│   │   ├── era5_1940_2026.grib            # Full meteorological reanalysis GRIB dataset
│   │   └── era5_monthly_1940_2026.grib.idx# GRIB binary index
│   └── processed/
│       └── era5_processed.csv             # Processed tabular dataset with coordinate keys
└── src/
    ├── notebooks/
    │   ├── 01_era5_data_audit.ipynb       # GRIB header validation & coverage checks
    │   ├── 02_preprocess_era5.ipynb       # Missing value handling, median imputation, CSV export
    │   ├── 03_baseline_aimri.ipynb        # Historical same-calendar-month 5-year baseline
    │   ├── 04_multiple_linear_regression.ipynb # Ridge regularized regression model pipeline
    │   ├── 05_convlstm_forecasting.ipynb  # Spatiotemporal ConvLSTM network implementation
    │   └── 06_model_comparison.ipynb      # 50-year common support benchmark & ranking engine
    └── artifacts/
        ├── aimri/                         # AIMRI configs, metric summaries, and spatial plots
        ├── audit/                         # Initial data audit reports and metadata logs
        ├── convlstm/                      # PyTorch weights (.pt), scalers, and tuning logs
        │   ├── models/mahamonsoon_convlstm.pt     # Serialized deep learning checkpoint
        │   ├── models/model_metadata.json # Architecture parameters and channel schemas
        │   └── results/test_metrics.csv   # Holdout test metrics
        ├── linear_regression/             # Scikit-learn joblib estimators and feature registries
        ├── model_comparison/              # Consolidated multi-model benchmark results
        │   ├── figures/                   # Visual comparison plots and error distributions
        │   └── results/
        │       ├── model_comparison_metrics.csv   # Unified metric table (MAE, RMSE, R²)
        │       ├── model_ranking.csv      # Official leaderboard ranking
        │       └── comparison_report.json # Full audit metadata & overlap report
        └── models/                        # Unified lightweight model deployment directory
```

---

## 🚀 Environment Setup & Execution Guide

### ⚠️ Python Version Requirement: `3.12.10`
This project relies on precompiled C-extensions for meteorological GRIB decoding (`eccodes`, `cfgrib`), PyTorch spatial tensor operations, and SciPy/Scikit-Learn. **Python 3.12.10 (64-bit)** is strictly required to prevent wheel compilation conflicts or binary incompatibilities.

---

### Option A: Automated 1-Click Setup (Recommended)

- **On Windows (Command Prompt / PowerShell):**
  Double-click or run:
  ```cmd
  setup_env.bat
  ```
  *(Checks for Python 3.12.10, creates the `.venv312` virtual environment, upgrades pip/setuptools/wheel, and installs all locked dependencies).*

- **On Linux / macOS / WSL:**
  ```bash
  chmod +x setup_env.sh
  ./setup_env.sh
  ```

---

### Option B: Manual Setup via Command Line

1. **Verify Python 3.12.10 is accessible:**
   ```bash
   # Windows (via Python Launcher)
   py -3.12 --version

   # Linux / macOS
   python3.12 --version
   ```

2. **Create and activate the virtual environment (`.venv312`):**
   ```bash
   # Windows
   py -3.12 -m venv .venv312
   .\.venv312\Scripts\activate

   # Linux / macOS
   python3.12 -m venv .venv312
   source .venv312/bin/activate
   ```

3. **Install locked dependencies:**
   ```bash
   python -m pip install --upgrade pip setuptools wheel
   pip install -r requirements.txt
   ```

4. **Register Jupyter Kernel (for notebook execution):**
   ```bash
   python -m ipykernel install --user --name=mahamonsoon-x --display-name "Python 3.12.10 (MahaMonsoon-X)"
   ```

---

### 💻 Step-by-Step Notebook Execution Workflow

Run the notebooks sequentially under `src/notebooks/` using the active `.venv312` environment:

```bash
# Step 1: Data Audit & Header Scan
jupyter execute src/notebooks/01_era5_data_audit.ipynb

# Step 2: Data Preprocessing & CSV Tabularization
jupyter execute src/notebooks/02_preprocess_era5.ipynb

# Step 3: Train & Evaluate Baseline AIMRI (Historical Mean)
jupyter execute src/notebooks/03_baseline_aimri.ipynb

# Step 4: Train & Evaluate Multiple Linear Regression (Ridge)
jupyter execute src/notebooks/04_multiple_linear_regression.ipynb

# Step 5: Train & Evaluate Spatiotemporal ConvLSTM
jupyter execute src/notebooks/05_convlstm_forecasting.ipynb

# Step 6: Execute Common-Support Benchmark & Generate Comparison Report
jupyter execute src/notebooks/06_model_comparison.ipynb
```

---

## 🔬 Related To

# **Seminar Research Paper**
```
  title   =  Spatiotemporal Deep Learning for Seasonal Monsoon Rainfall Prediction over Maharashtra
  author  = Shantanu Suryawanshi
  Guide   = Prakash Ukhalkar 
  College = Pimpri Chinchwad College Of Engineering (PCCOE) , Pune
  Dept    = MCA 
  Year    = 2025 - 2026
  sem     = 3 

```

---

<p align="center">
  <b>MahaMonsoon-X</b> • Advancing Regional Hydroclimatic Science through Open, Reproducible AI Benchmarking.
</p>
