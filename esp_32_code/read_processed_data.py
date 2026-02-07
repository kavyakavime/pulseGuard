"""
PulseGuard - Live PPG Monitor with Signal Processing
====================================================
Uses the full signal processing pipeline:
1. DC removal
2. Band-pass filter (0.5-5 Hz)
3. Peak detection
4. HR, HRV (RMSSD, SDNN) from inter-beat intervals
"""

import serial
import serial.tools.list_ports
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from collections import deque

from ppg_processor import (
    process_ppg_pipeline,
    remove_dc,
    bandpass_filter,
    detect_peaks,
)

BAUD = 115200
FS_RAW = 100.0   # Sample rate when getting raw (3-part) lines
FS_CSV = 10.0    # Sample rate when getting CSV (10-part) lines only
WINDOW_SEC = 10
MIN_SAMPLES_RAW = int(2 * FS_RAW)   # 200 for 100 Hz
MIN_SAMPLES_CSV = int(2 * FS_CSV)   # 20 for 10 Hz

# Buffers for raw IR (max 1000 samples)
time_buffer = deque(maxlen=1000)
ir_buffer = deque(maxlen=1000)

# Processed results
hr_history = deque(maxlen=300)
hrv_rmssd_history = deque(maxlen=300)
hrv_sdnn_history = deque(maxlen=300)
ibi_history = deque(maxlen=300)
result_time = deque(maxlen=300)

# Hold last good values when signal is bad
last_hr = 0.0
last_hrv = 0.0
last_ibi = 0.0

def find_serial_port():
    """Find Arduino/serial port - prefer usbserial, fallback to first available."""
    ports = list(serial.tools.list_ports.comports())
    usb = [p.device for p in ports if "usbserial" in p.device.lower() or "usbmodem" in p.device.lower()]
    if usb:
        return usb[0]
    if ports:
        return ports[0].device
    return "/dev/cu.usbserial-0001"

print("PulseGuard - Signal Processing Monitor")
print("=" * 50)
print("Pipeline: DC removal -> Band-pass 0.5-5 Hz -> Peak detection -> HR/HRV")

PORT = find_serial_port()
print(f"Connecting to {PORT}...")

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.01)
except serial.SerialException as e:
    print(f"\nSerial error: {e}")
    print("Available ports:", [p.device for p in serial.tools.list_ports.comports()])
    exit(1)

time.sleep(2)

# Flush startup junk
for _ in range(20):
    ser.readline()

print("Connected. Reading data...\n")

# Setup plots
plt.ion()
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Raw PPG
ax_raw = fig.add_subplot(gs[0, :])
line_raw, = ax_raw.plot([], [], 'r-', label='IR (raw)', linewidth=1, alpha=0.8)
ax_raw.set_title('Raw PPG Signal', fontsize=12, fontweight='bold')
ax_raw.set_ylabel('ADC Value')
ax_raw.legend(loc='upper right')
ax_raw.grid(True, alpha=0.3)

# Filtered PPG with peaks
ax_filt = fig.add_subplot(gs[1, :])
line_filt, = ax_filt.plot([], [], 'b-', label='Filtered', linewidth=1)
line_peaks, = ax_filt.plot([], [], 'ro', markersize=4, label='Beats')
ax_filt.set_title('Filtered PPG (0.5-5 Hz) + Detected Beats', fontweight='bold')
ax_filt.set_ylabel('Amplitude')
ax_filt.legend(loc='upper right')
ax_filt.grid(True, alpha=0.3)

# HR
ax_hr = fig.add_subplot(gs[2, 0])
line_hr, = ax_hr.plot([], [], 'crimson', linewidth=2)
ax_hr.set_title('Heart Rate (from peaks)', fontweight='bold')
ax_hr.set_ylabel('BPM')
ax_hr.set_ylim(40, 120)
ax_hr.grid(True, alpha=0.3)
ax_hr.axhspan(60, 100, alpha=0.1, color='green')

# HRV
ax_hrv = fig.add_subplot(gs[2, 1])
line_rmssd, = ax_hrv.plot([], [], 'purple', linewidth=2, label='RMSSD')
line_sdnn, = ax_hrv.plot([], [], 'green', linewidth=2, label='SDNN')
ax_hrv.set_title('HRV (RMSSD / SDNN)', fontweight='bold')
ax_hrv.set_ylabel('ms')
ax_hrv.legend(loc='upper right')
ax_hrv.set_ylim(0, 200)
ax_hrv.grid(True, alpha=0.3)

# IBI
ax_ibi = fig.add_subplot(gs[3, :])
line_ibi, = ax_ibi.plot([], [], 'orange', linewidth=2, marker='o', markersize=3)
ax_ibi.set_title('Inter-Beat Interval', fontweight='bold')
ax_ibi.set_xlabel('Time (seconds)')
ax_ibi.set_ylabel('IBI (ms)')
ax_ibi.set_ylim(300, 1500)
ax_ibi.grid(True, alpha=0.3)

fig.canvas.draw()

print("Window opened. Monitoring...\n")

try:
    last_process_time = 0
    samples_per_sec = 0
    last_sample_count = 0
    last_rate_time = time.time()
    
    while True:
        # Read serial - accept BOTH 3-part (raw) and 10-part (CSV) - use first 3 values
        for _ in range(200):
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("ERROR") or line.startswith("["):
                    continue
                parts = line.split(",")
                if len(parts) < 3:
                    continue
                timestamp = int(parts[0])
                ir = int(parts[1])
                red = int(parts[2])
                
                time_buffer.append(timestamp / 1000.0)
                ir_buffer.append(ir)
            except (ValueError, IndexError):
                continue
        
        # Infer sample rate from data
        n = len(ir_buffer)
        if n >= 2:
            t_span = time_buffer[-1] - time_buffer[0]
            if t_span > 0.1:
                FS = (n - 1) / t_span
            else:
                FS = FS_RAW if n > 50 else FS_CSV
        else:
            FS = FS_CSV
        MIN_SAMPLES = int(2 * FS)
        
        # Always update raw PPG when we have data
        if len(ir_buffer) > 1:
            t_arr = np.array(list(time_buffer))
            ir_arr = np.array(list(ir_buffer))
            line_raw.set_data(t_arr, ir_arr)
            ax_raw.set_xlim(t_arr[0], t_arr[-1])
            if len(ir_arr) > 0:
                r = np.ptp(ir_arr) or 1000
                y_min = np.min(ir_arr) - 0.05 * r
                y_max = np.max(ir_arr) + 0.05 * r
                ax_raw.set_ylim(y_min, y_max)
        
        # Run full processing every 500ms when we have enough samples
        now = time.time()
        if now - last_process_time >= 0.5 and len(ir_buffer) >= MIN_SAMPLES:
            last_process_time = now
            
            t = np.array(list(time_buffer))
            ir = np.array(list(ir_buffer))
            
            result = process_ppg_pipeline(ir, fs=FS)
            
            # Use result if good (40-120 BPM, quality>=30); else hold last good
            hr_val = result["hr"]
            quality = result.get("quality", 0)
            if hr_val >= 40 and hr_val <= 120 and quality >= 30:
                last_hr = hr_val
                last_hrv = result["hrv_rmssd"]
                last_ibi = np.median(result["ibi_ms"]) if len(result["ibi_ms"]) > 0 else last_ibi
            elif last_hr > 0:
                hr_val = last_hr
                result = {**result, "hr": last_hr, "hrv_rmssd": last_hrv}
            
            if hr_val > 0:
                hr_history.append(hr_val)
                hrv_rmssd_history.append(result["hrv_rmssd"])
                hrv_sdnn_history.append(result["hrv_sdnn"])
                result_time.append(t[-1])
                ibi_mean = np.median(result["ibi_ms"]) if len(result["ibi_ms"]) > 0 else last_ibi
                if ibi_mean > 0:
                    last_ibi = ibi_mean
                ibi_history.append(ibi_mean if ibi_mean > 0 else last_ibi)
            
            # Update plots (raw PPG already updated above)
            t_arr = np.array(t)
            
            # Filtered + peaks
            filt = result["filtered_signal"]
            peaks = result["peaks"]
            if len(filt) > 0:
                line_filt.set_data(t_arr[:len(filt)], filt)
                if len(peaks) > 0:
                    line_peaks.set_data(t_arr[peaks], filt[peaks])
                else:
                    line_peaks.set_data([], [])
                ax_filt.set_xlim(t_arr[0], t_arr[-1])
                if len(filt) > 0:
                    r = np.ptp(filt)
                    ax_filt.set_ylim(np.min(filt) - 0.1 * r, np.max(filt) + 0.1 * r)
            
            # HR
            if len(hr_history) > 0:
                tr = list(result_time)
                line_hr.set_data(tr, list(hr_history))
                ax_hr.set_xlim(tr[0], tr[-1])
            
            # HRV
            if len(hrv_rmssd_history) > 0:
                tr = list(result_time)
                line_rmssd.set_data(tr, list(hrv_rmssd_history))
                line_sdnn.set_data(tr, list(hrv_sdnn_history))
                ax_hrv.set_xlim(tr[0], tr[-1])
            
            # IBI
            if len(ibi_history) > 0:
                tr = list(result_time)
                line_ibi.set_data(tr, list(ibi_history))
                ax_ibi.set_xlim(tr[0], tr[-1])
            
            # Terminal output
            n_buf = len(ir_buffer)
            q = result.get("quality", 0)
            if result["hr"] > 0:
                hold = " (holding)" if q < 30 else ""
                print(f"\r[{t_arr[-1]:.1f}s] HR: {result['hr']:.1f} BPM | "
                      f"HRV: {result['hrv_rmssd']:.1f} ms | "
                      f"IBI: {last_ibi:.0f} ms | "
                      f"Q:{q:.0f}{hold}", end="", flush=True)
            else:
                print(f"\r[{t_arr[-1]:.1f}s] No beats | "
                      f"Buffer: {n_buf} | "
                      f"Hold finger steady on sensor", end="", flush=True)
        elif len(ir_buffer) > 0:
            print(f"\rBuffer: {len(ir_buffer)} samples (need {MIN_SAMPLES})...", end="", flush=True)
        
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.05)

except KeyboardInterrupt:
    print("\n\nMonitoring stopped")
except Exception as e:
    print(f"\nError: {e}")
finally:
    try:
        ser.close()
    except Exception:
        pass
    try:
        plt.ioff()
        plt.close()
    except Exception:
        pass
    print("Serial connection closed.")
