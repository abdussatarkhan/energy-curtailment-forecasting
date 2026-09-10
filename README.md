# Renewable Energy Curtailment & Grid Stress Forecaster

[![PyTorch](https://img.shields.io/badge/PyTorch-LSTM_Ensemble-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org/) [![Prophet](https://img.shields.io/badge/Time_Series-Prophet-00A8E8?style=for-the-badge)](https://facebook.github.io/prophet/) [![Python](https://img.shields.io/badge/Python-Energy_Analytics-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Author](https://img.shields.io/badge/Author-Abdussatar-E50914?style=for-the-badge&logo=github&logoColor=white)](https://github.com/abdussatarkhan)

> **A hybrid time-series forecasting system combining Facebook Prophet and Deep LSTM neural networks to predict renewable energy (Solar/Wind) curtailment and transmission bottlenecks up to 72 hours ahead.**

---

## 🏛️ System Architecture

```mermaid
graph TD
    Weather[NWP Weather & Solar/Wind Gen Telemetry] --> Feature[Time-Lagged Feature Engineering]
    Feature --> Prophet[Facebook Prophet Trend Decomposition]
    Feature --> LSTM[Deep PyTorch LSTM Network]
    Prophet --> Stacking[Ridge Stacking Ensemble]
    LSTM --> Stacking
    Stacking --> GridSchedule[72-Hour Curtailment Schedule]
```

---

## 🌟 Key Features & Capabilities

- **Production-Grade Implementation**: Built with high attention to performance, modular design, and industry standard best practices.
- **Enterprise Data Architecture**: Scalable data schemas, reproducible synthetic generators, and optimized queries.
- **Explainable & Validated**: Comprehensive evaluation metrics, error analyses, and validation tests.
- **Comprehensive Tech Stack**: `Python` `PyTorch` `Prophet` `NumPy` `Pandas` `Time Series`.

---

## 📊 Visual Preview & Analysis

<div align="center">

![energy-curtailment-forecasting preview](images/prophet_lstm_forecast.png)

</div>

---

## 🚀 Quickstart & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/abdussatarkhan/energy-curtailment-forecasting.git
cd energy-curtailment-forecasting
```

### 2. Environment Setup
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate

# Install dependencies (if requirements.txt exists)
pip install -r requirements.txt
```

---

## 👨‍💻 Author & Profile

Built and maintained by **Abdussatar** ([@abdussatarkhan](https://github.com/abdussatarkhan)).  
For technical discussions, collaboration, or queries, feel free to reach out via [LinkedIn](https://www.linkedin.com/in/abdus-satar-5150813b5/) or [GitHub](https://github.com/abdussatarkhan).

---

## 📜 License

This project is licensed under the **MIT License** — see the LICENSE file for details.
