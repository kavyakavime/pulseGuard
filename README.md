# PulseGuard  
### Real-Time Physiological Strain Detection with Low-Cost Biosensors

PulseGuard is a **real-time community health and sustainability platform** that uses low-cost biosensors and machine learning to detect early signs of **physiological stress and system strain**.

By capturing **photoplethysmography (PPG)** signals through optical pulse sensors such as the **MAX30100 / MAX30102**, PulseGuard measures **heart rate and heart rate variability (HRV)**—key indicators of fatigue, stress, and irregular workload. The system is designed to operate **without specialized medical equipment**, making continuous monitoring accessible to **underserved or high-risk communities**.

> ⚠️ Disclaimer: PulseGuard is **not a medical diagnostic device**. It is intended for research, education, and early-warning insights only.

---

## 🧠 Core Concept

PulseGuard treats **human capacity as a measurable and protectable resource**.

Instead of relying on rigid medical thresholds, the system:
- Learns a baseline of *normal physiological operation*
- Detects deviations that indicate rising strain
- Surfaces insights early—before critical failure or burnout occurs

This approach enables **preventive awareness**, not reactive diagnosis.

---

## 🏗️ System Architecture
PPG Sensor → ESP32 → Raspberry Pi →
Signal Processing → Feature Extraction →
ML-Based Anomaly Detection → Web Dashboard


### Data Flow
1. **PPG Acquisition**  
   Optical pulse sensors capture raw blood volume signals.

2. **Edge Streaming**  
   An ESP32 streams sensor data via serial to a Raspberry Pi.

3. **Signal Processing**  
   Noise filtering, peak detection, and temporal normalization are applied.

4. **Feature Extraction**  
   Heart rate, HRV, and derived strain indicators are computed.

5. **Machine Learning**  
   Models learn baseline behavior and flag anomalous strain patterns.

6. **Visualization**  
   A web dashboard presents real-time wellness and strain events.

---

## 🛠️ Hardware Components

| Component | Purpose |
|---------|--------|
| ESP32 | Data acquisition and streaming |
| MAX30100 / MAX30102 | PPG-based heart rate sensing |
| Raspberry Pi | Signal processing, ML inference, dashboard hosting |
| Power Supply | Low-voltage regulated input |

The hardware stack is intentionally **minimal and affordable**, enabling deployment in non-clinical environments.

---

## 💻 Software Stack

- **Languages**
  - Python (signal processing, ML, visualization)
  - Arduino/C++ (ESP32 firmware)

- **Core Techniques**
  - Digital signal filtering
  - HR & HRV feature extraction
  - Baseline modeling
  - Anomaly detection

- **Visualization**
  - Web-based dashboard for real-time insights


---

## 🤖 Machine Learning Approach

PulseGuard uses **ground-up ML models** trained on physiological time-series data.

### Key Ideas
- Learn what *normal* looks like for an individual or group
- Detect deviations rather than classify diseases
- Emphasize **early signals** of overload or instability

This enables:
- Individual burnout awareness
- Collective system strain detection
- Explainable, actionable alerts

---

## 🌍 Why PulseGuard Matters

- **Accessible**: No clinical equipment required
- **Preventive**: Detects strain before failure
- **Scalable**: Suitable for individuals, teams, or communities
- **Explainable**: Focuses on interpretable physiological signals
- **Sustainable**: Protects human capacity as a shared resource

PulseGuard transforms **invisible stress into actionable foresight**, helping communities stay resilient, balanced, and connected.

---

## 🚧 Future Directions

- Wireless streaming (WiFi / BLE)
- On-device inference (TinyML)
- Multi-user aggregation for community health insights
- Integration with sustainability and workload metrics
- Long-term trend analysis and adaptive baselining

---

## 📜 License
MIT License


