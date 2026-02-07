import serial
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import numpy as np
from collections import deque

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200
BUFFER_SIZE = 300  # Show last 30 seconds (at 10Hz display rate)

# Data buffers
time_buffer = deque(maxlen=BUFFER_SIZE)
ir_buffer = deque(maxlen=BUFFER_SIZE)
red_buffer = deque(maxlen=BUFFER_SIZE)
bpm_buffer = deque(maxlen=BUFFER_SIZE)
spo2_buffer = deque(maxlen=BUFFER_SIZE)
hrv_buffer = deque(maxlen=BUFFER_SIZE)
ibi_buffer = deque(maxlen=BUFFER_SIZE)
quality_buffer = deque(maxlen=BUFFER_SIZE)

print("🫀 PulseGuard - Live Monitor + Plot")
print("=" * 50)
print(f"Connecting to {PORT}...")

ser = serial.Serial(PORT, BAUD, timeout=0.01)
time.sleep(2)

# Skip header
header = ser.readline().decode("utf-8", errors="ignore").strip()
print("✓ Connected! Starting live monitoring...\n")

# Setup plots
plt.ion()
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

# Raw PPG (top, full width)
ax_ppg = fig.add_subplot(gs[0, :])
line_ir, = ax_ppg.plot([], [], 'r-', label='IR', linewidth=1, alpha=0.8)
line_red, = ax_ppg.plot([], [], 'darkred', label='Red', linewidth=1, alpha=0.6)
ax_ppg.set_title('Raw PPG Signal', fontsize=12, fontweight='bold')
ax_ppg.set_ylabel('ADC Value')
ax_ppg.legend(loc='upper right')
ax_ppg.grid(True, alpha=0.3)

# BPM
ax_bpm = fig.add_subplot(gs[1, 0])
line_bpm, = ax_bpm.plot([], [], 'crimson', linewidth=2, marker='o', markersize=3)
ax_bpm.set_title('Heart Rate', fontweight='bold')
ax_bpm.set_ylabel('BPM')
ax_bpm.set_ylim(40, 120)
ax_bpm.grid(True, alpha=0.3)
ax_bpm.axhspan(60, 100, alpha=0.1, color='green')

# SpO2
ax_spo2 = fig.add_subplot(gs[1, 1])
line_spo2, = ax_spo2.plot([], [], 'royalblue', linewidth=2)
ax_spo2.set_title('Blood Oxygen (SpO2)', fontweight='bold')
ax_spo2.set_ylabel('SpO₂ (%)')
ax_spo2.set_ylim(85, 102)
ax_spo2.grid(True, alpha=0.3)
ax_spo2.axhspan(95, 100, alpha=0.1, color='green')

# HRV
ax_hrv = fig.add_subplot(gs[2, 0])
line_hrv, = ax_hrv.plot([], [], 'purple', linewidth=2, marker='o', markersize=3)
ax_hrv.set_title('Heart Rate Variability (RMSSD)', fontweight='bold')
ax_hrv.set_ylabel('HRV (ms)')
ax_hrv.set_ylim(0, 200)
ax_hrv.grid(True, alpha=0.3)

# IBI
ax_ibi = fig.add_subplot(gs[2, 1])
line_ibi, = ax_ibi.plot([], [], 'orange', linewidth=2, marker='o', markersize=3)
ax_ibi.set_title('Inter-Beat Interval', fontweight='bold')
ax_ibi.set_ylabel('IBI (ms)')
ax_ibi.set_ylim(300, 2000)
ax_ibi.grid(True, alpha=0.3)

# Signal Quality
ax_quality = fig.add_subplot(gs[3, :])
line_quality, = ax_quality.plot([], [], 'green', linewidth=2, alpha=0.7)
ax_quality.set_title('Signal Quality', fontweight='bold')
ax_quality.set_xlabel('Time (seconds)')
ax_quality.set_ylabel('Quality (%)')
ax_quality.set_ylim(0, 110)
ax_quality.grid(True, alpha=0.3)
ax_quality.axhspan(80, 100, alpha=0.1, color='green')

plt.tight_layout()
fig.canvas.draw()

print("Window opened. Monitoring...\n")

try:
    frame_count = 0
    
    while True:
        # Read serial data
        for _ in range(10):
            try:
                line = ser.readline().decode("utf-8", errors="ignore").strip()
                if not line or line.startswith("ERROR") or not line[0].isdigit():
                    continue
                parts = line.split(",")
                    
                if len(parts) >= 9:
                        timestamp = int(parts[0])
                        ir = int(parts[1])
                        red = int(parts[2])
                        bpm = float(parts[3])
                        hrv = float(parts[4])
                        spo2 = float(parts[5])
                        
                        if len(parts) == 10:
                            ibi = float(parts[6])
                            finger_detected = int(parts[7])
                            hrv_ready = int(parts[8])
                            beat_quality = float(parts[9])
                        else:
                            ibi = 0
                            finger_detected = int(parts[6])
                            hrv_ready = int(parts[7])
                            beat_quality = float(parts[8])
                        
                        # Add to buffers
                        time_buffer.append(timestamp / 1000.0)
                        ir_buffer.append(ir)
                        red_buffer.append(red)
                        bpm_buffer.append(bpm if bpm > 0 else None)
                        spo2_buffer.append(spo2 if spo2 > 0 else None)
                        hrv_buffer.append(hrv if hrv > 0 else None)
                        ibi_buffer.append(ibi if ibi > 0 else None)
                        quality_buffer.append(beat_quality)
                        
                        # Terminal output (every 10 samples)
                        if frame_count % 10 == 0:
                            print(f"\r[{timestamp/1000:.1f}s] ", end="")
                            
                            if finger_detected:
                                print(f"💓 BPM: {bpm:5.1f} | 🫁 SpO2: {spo2:5.1f}% | ", end="")
                                if ibi > 0:
                                    print(f"⏱️ IBI: {ibi:.0f}ms | ", end="")
                                if hrv_ready:
                                    print(f"📊 HRV: {hrv:5.1f}ms | ", end="")
                                
                                quality_bars = int(beat_quality / 10)
                                quality_str = "█" * quality_bars + "░" * (10 - quality_bars)
                                print(f"Signal: {quality_str}", end="")
                            else:
                                print("👆 No finger detected", end="")
                            
                            print("", flush=True)
                        
                        frame_count += 1
            except (ValueError, IndexError):
                continue
        
        # Update plots (every 50ms)
        if len(time_buffer) > 1:
            t = list(time_buffer)
            
            # Raw PPG
            line_ir.set_data(t, list(ir_buffer))
            line_red.set_data(t, list(red_buffer))
            ax_ppg.set_xlim(t[0], t[-1])
            if len(ir_buffer) > 0:
                y_min = min(min(ir_buffer), min(red_buffer)) * 0.98
                y_max = max(max(ir_buffer), max(red_buffer)) * 1.02
                ax_ppg.set_ylim(y_min, y_max)
            
            # BPM
            bpm_clean = [v for v in bpm_buffer if v is not None]
            if len(bpm_clean) > 0:
                t_bpm = [t[i] for i in range(len(bpm_buffer)) if bpm_buffer[i] is not None]
                line_bpm.set_data(t_bpm, bpm_clean)
                ax_bpm.set_xlim(t[0], t[-1])
            
            # SpO2
            spo2_clean = [v for v in spo2_buffer if v is not None]
            if len(spo2_clean) > 0:
                t_spo2 = [t[i] for i in range(len(spo2_buffer)) if spo2_buffer[i] is not None]
                line_spo2.set_data(t_spo2, spo2_clean)
                ax_spo2.set_xlim(t[0], t[-1])
            
            # HRV
            hrv_clean = [v for v in hrv_buffer if v is not None]
            if len(hrv_clean) > 0:
                t_hrv = [t[i] for i in range(len(hrv_buffer)) if hrv_buffer[i] is not None]
                line_hrv.set_data(t_hrv, hrv_clean)
                ax_hrv.set_xlim(t[0], t[-1])
            
            # IBI
            ibi_clean = [v for v in ibi_buffer if v is not None]
            if len(ibi_clean) > 0:
                t_ibi = [t[i] for i in range(len(ibi_buffer)) if ibi_buffer[i] is not None]
                line_ibi.set_data(t_ibi, ibi_clean)
                ax_ibi.set_xlim(t[0], t[-1])
            
            # Signal Quality
            line_quality.set_data(t, list(quality_buffer))
            ax_quality.set_xlim(t[0], t[-1])
        
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
