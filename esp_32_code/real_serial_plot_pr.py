"""
PulseGuard - Live Monitor
=========================
Raw PPG from 3-part stream (100 Hz) for beat detection.
HR/HRV/IBI from 10-part stream (Arduino vitals).
Pipeline: Raw -> DC removal -> Median filter -> Normalize -> Peak detection -> Beats
(Last 6 sec only, demo-proof)
"""

import serial
import serial.tools.list_ports
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from collections import deque
from scipy import signal as scipy_signal

from ppg_processor import remove_dc
from ppg_features import StrainMonitor

# 3-part (time,ir,red) at 100 Hz for beat detection. 10-part at 10 Hz for vitals.
FS_PPG = 100.0   # Sampling rate for raw PPG (beat detection)
FS_VITALS = 10.0 # Arduino vitals update rate

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200

t_buf = deque(maxlen=1200)   # 12 sec at 100 Hz
ir_buf = deque(maxlen=1200)
bpm_buf = deque(maxlen=1200)
hrv_buf = deque(maxlen=1200)
ibi_buf = deque(maxlen=1200)
spo2_buf = deque(maxlen=1200)

arduino_bpm_last = 0
arduino_hrv_last = 0
arduino_ibi_last = 0
arduino_spo2_last = 0

strain_monitor = StrainMonitor(window_sec=60, baseline_sec=30)
strain_history = deque(maxlen=600)
feature_time = deque(maxlen=600)

def find_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "usbserial" in p.device.lower() or "usbmodem" in p.device.lower():
            return p.device
    return ports[0].device if ports else PORT

print("PulseGuard - Live Monitor")
print("=" * 40)

try:
    p = find_port()
    ser = serial.Serial(p, BAUD, timeout=0.02)
except Exception as e:
    print(f"Serial error: {e}")
    print("Try: ls /dev/cu.*")
    exit(1)

time.sleep(2)
ser.reset_input_buffer()

plt.ion()
fig, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=True)
ax_raw, ax_filt, ax_hr, ax_hrv, ax_ibi, ax_strain = axes

line_raw, = ax_raw.plot([], [], 'r-', lw=1)
ax_raw.set_ylabel("AC (DC-removed)")
ax_raw.set_title("Raw PPG (DC-removed)")
ax_raw.grid(True, alpha=0.3)

line_filt, = ax_filt.plot([], [], 'b-', lw=1, label='Filtered')
line_beats, = ax_filt.plot([], [], 'ro', markersize=8, label='Beats')
ax_filt.set_ylabel("Amplitude")
ax_filt.set_title("Filtered PPG (median) + Detected Beats")
ax_filt.legend(loc='upper right')
ax_filt.grid(True, alpha=0.3)

line_hr, = ax_hr.plot([], [], 'crimson', lw=2)
ax_hr.set_ylabel("BPM")
ax_hr.set_title("Heart Rate")
ax_hr.set_ylim(40, 120)
ax_hr.grid(True, alpha=0.3)

line_hrv, = ax_hrv.plot([], [], 'purple', lw=2)
ax_hrv.set_ylabel("ms")
ax_hrv.set_title("HRV (RMSSD)")
ax_hrv.set_ylim(0, 200)
ax_hrv.grid(True, alpha=0.3)

line_ibi, = ax_ibi.plot([], [], 'orange', lw=2)
ax_ibi.set_ylabel("ms")
ax_ibi.set_title("Inter-Beat Interval")
ax_ibi.set_ylim(400, 1500)
ax_ibi.grid(True, alpha=0.3)

line_strain, = ax_strain.plot([], [], 'red', lw=3)
ax_strain.set_ylabel("Strain (0-1)")
ax_strain.set_xlabel("Time (s)")
ax_strain.set_title("Strain Index")
ax_strain.set_ylim(0, 1)
ax_strain.axhspan(0.0, 0.3, alpha=0.1, color='green')
ax_strain.axhspan(0.3, 0.6, alpha=0.1, color='yellow')
ax_strain.axhspan(0.6, 1.0, alpha=0.1, color='red')
ax_strain.grid(True, alpha=0.3)

plt.tight_layout()
fig.canvas.draw()

print("Connected. Place finger on sensor (100 Hz PPG for beat detection).\n")

try:
    while True:
        # Read ALL available lines - need 3-part for 100 Hz PPG
        for _ in range(200):
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("ERROR"):
                    continue
                if line.startswith("["):  # Skip human-readable
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue

                ts = int(parts[0]) / 1000.0
                ir = int(parts[1])

                # 3-part: time,ir,red - add to PPG buffers (100 Hz)
                t_buf.append(ts)
                ir_buf.append(ir)
                bpm_buf.append(arduino_bpm_last)
                hrv_buf.append(arduino_hrv_last)
                ibi_buf.append(arduino_ibi_last)

                # 9-part: time,ir,red,bpm,hrv,spo2,fingerDetected,hrvReady,beatQuality
                if len(parts) >= 9:
                    arduino_bpm_last = float(parts[3])
                    arduino_hrv_last = float(parts[4])
                    spo2_raw = float(parts[5])
                    # Add 85 to SpO2 for display (Arduino sends raw value)
                    arduino_spo2_last = min(spo2_raw + 85, 100) if spo2_raw > 0 else 0
                    # IBI can be calculated from HRV or BPM if needed
                    arduino_ibi_last = (60000.0 / arduino_bpm_last) if arduino_bpm_last > 0 else 0
                    spo2_buf.append(arduino_spo2_last)

            except (ValueError, IndexError):
                continue

        # Update plots
        if len(t_buf) > 2:
            t = np.array(t_buf)
            ir_arr = np.array(list(ir_buf)).astype(float)
            ir_arr = ir_arr - np.mean(ir_arr)  # Quick DC removal for display

            line_raw.set_data(t, ir_arr)
            ax_raw.set_xlim(t[0], t[-1])
            if np.ptp(ir_arr) > 10:
                margin = np.ptp(ir_arr) * 0.15
                ax_raw.set_ylim(np.min(ir_arr) - margin, np.max(ir_arr) + margin)
            else:
                ax_raw.set_ylim(-500, 500)

            num_beats = 0
            N = int(6 * FS_PPG)  # Last 6 seconds only
            if len(ir_arr) >= N:
                try:
                    peaks = np.array([])
                    sig = ir_arr[-N:].astype(float)
                    dc_removed = remove_dc(sig, method="ema")
                    filtered = scipy_signal.medfilt(dc_removed, kernel_size=3)  # 3 = less smoothing, preserves peaks

                    # Normalize: DC remove, scale to ~±1
                    f = filtered - np.mean(filtered)
                    std = np.std(f)
                    if std < 1e-6:
                        peaks = np.array([])
                    else:
                        f = f / std
                        dist = max(1, int(0.4 * FS_PPG))  # 400ms = 150 BPM max
                        # PPG: systolic = peak or valley. Try both, use whichever finds beats.
                        p_pos, _ = scipy_signal.find_peaks(f, distance=dist, height=0.05)
                        p_neg, _ = scipy_signal.find_peaks(-f, distance=dist, height=0.05)
                        peaks = p_pos if len(p_pos) >= len(p_neg) else p_neg
                        # Fallback: relax height if nothing found
                        if len(peaks) < 2 and np.ptp(f) > 0.5:
                            p_pos, _ = scipy_signal.find_peaks(f, distance=dist, height=0.02)
                            p_neg, _ = scipy_signal.find_peaks(-f, distance=dist, height=0.02)
                            peaks = p_pos if len(p_pos) >= len(p_neg) else p_neg
                        peaks = peaks + (len(ir_arr) - N)  # Map back to full buffer

                    line_filt.set_data(t[-N:], filtered)
                    if len(peaks) > 0:
                        offset = len(ir_arr) - N
                        valid_idx = peaks[(peaks < len(t)) & (peaks >= offset) & (peaks < len(ir_arr))]
                        if len(valid_idx) > 0:
                            peak_in_seg = valid_idx - offset
                            num_beats = len(valid_idx)
                            line_beats.set_data(t[valid_idx], filtered[peak_in_seg])
                        else:
                            line_beats.set_data([], [])
                    else:
                        line_beats.set_data([], [])

                    ax_filt.set_xlim(t[-N], t[-1])
                    if np.ptp(filtered) > 1:
                        margin = np.ptp(filtered) * 0.2
                        ax_filt.set_ylim(np.min(filtered) - margin, np.max(filtered) + margin)
                    else:
                        ax_filt.set_ylim(-500, 500)

                except Exception:
                    line_filt.set_data(t, np.zeros_like(t))
                    line_beats.set_data([], [])

            line_hr.set_data(t, list(bpm_buf))
            line_hrv.set_data(t, list(hrv_buf))
            line_ibi.set_data(t, list(ibi_buf))
            ax_hr.set_xlim(t[0], t[-1])
            ax_hrv.set_xlim(t[0], t[-1])
            ax_ibi.set_xlim(t[0], t[-1])

            display_hr = arduino_bpm_last if arduino_bpm_last > 0 else (bpm_buf[-1] if bpm_buf else 0)
            display_hrv = arduino_hrv_last
            display_ibi = arduino_ibi_last
            if display_hr > 0:
                strain_monitor.add_sample(t[-1], display_hr, display_hrv, display_ibi)

            features = strain_monitor.get_features()
            strain_history.append(features["strain_index"])
            feature_time.append(t[-1])
            if len(strain_history) > 1:
                ft = np.array(feature_time)
                line_strain.set_data(ft, list(strain_history))
                ax_strain.set_xlim(ft[0], ft[-1])

            status = "STRAINED" if features["strain_index"] > 0.6 else ("MODERATE" if features["strain_index"] > 0.3 else "RELAXED")
            base = strain_monitor.get_baseline_str()
            print(f"\r[{t[-1]:.0f}s] HR: {display_hr:.1f} | HRV: {display_hrv:.1f}ms | IBI: {display_ibi:.0f}ms | "
                  f"Beats: {num_beats} | Strain: {features['strain_index']:.2f} ({status})", end="", flush=True)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.02)

except KeyboardInterrupt:
    print("\n\nStopped.")
finally:
    ser.close()
    plt.ioff()
    plt.close()
    print("Done.")
