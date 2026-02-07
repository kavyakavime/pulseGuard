import serial
import time
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np
from datetime import datetime
import os

PORT = "/dev/cu.usbserial-0001"  # Update for Pi: /dev/ttyUSB0 or /dev/ttyACM0
BAUD = 115200

# CSV settings - ML training format
SAVE_CHUNK_SIZE = 1000  # Save every 1000 samples
SAVE_DIR = "ppg_data"
CSV_HEADER = "time,ir,red,bpm,hrv,spo2,fingerDetected,hrvReady,beatQuality"

os.makedirs(SAVE_DIR, exist_ok=True)

BUFFER_SIZE = 500
time_buffer = deque(maxlen=BUFFER_SIZE)
ir_buffer = deque(maxlen=BUFFER_SIZE)
red_buffer = deque(maxlen=BUFFER_SIZE)

csv_buffer = []
chunk_counter = 0

current_vitals = {
    'bpm': 0, 'spo2': 0, 'hrv': 0, 'ibi': 0,
    'quality': 0, 'finger': False
}
bpm_smooth = 0.0  # EMA for stable display

print("🫀 PulseGuard - Live PPG Monitor + CSV Saver")
print("=" * 50)
print(f"Connecting to {PORT}...")
print(f"Saving to: {SAVE_DIR}/")

try:
    ser = serial.Serial(PORT, BAUD, timeout=0.1)
    time.sleep(2)
    
    ser.reset_input_buffer()
    print("✓ Connected! Saving in ML format (time,ir,red,bpm,hrv,spo2,fingerDetected,hrvReady,beatQuality)\n")
    
    # Setup plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('🫀 PulseGuard - Live PPG', fontsize=16, fontweight='bold')
    
    line_ir, = ax1.plot([], [], 'r-', label='IR', linewidth=1.5, alpha=0.8)
    line_red, = ax1.plot([], [], 'darkred', label='Red', linewidth=1.5, alpha=0.6)
    ax1.set_ylabel('ADC Value')
    ax1.set_title('Photoplethysmogram (PPG)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    line_ir_norm, = ax2.plot([], [], 'r-', label='IR (norm)', linewidth=1.5)
    line_red_norm, = ax2.plot([], [], 'darkred', label='Red (norm)', linewidth=1.5, alpha=0.6)
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Normalized')
    ax2.set_title('Normalized PPG')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-2, 2)
    
    vitals_text = fig.text(0.02, 0.02, '', fontsize=9, family='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    def save_csv_chunk():
        """Save buffered data to CSV (ML training format)"""
        global chunk_counter, csv_buffer
        
        if len(csv_buffer) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{SAVE_DIR}/ppg_chunk_{timestamp}_{chunk_counter:04d}.csv"
            
            with open(filename, 'w') as f:
                f.write(CSV_HEADER + "\n")
                for row in csv_buffer:
                    f.write(row + "\n")
            
            print(f"💾 Saved {len(csv_buffer)} samples to {filename}")
            csv_buffer = []
            chunk_counter += 1
    
    def read_serial():
        """Read serial and update buffers. Save in ML format: time,ir,red,bpm,hrv,spo2,fingerDetected,hrvReady,beatQuality"""
        global csv_buffer, bpm_smooth
        
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
            
            if line and not line.startswith("ERROR") and not line.startswith("["):
                parts = line.split(",")
                
                if len(parts) >= 9:
                    # Arduino: time,ir,red,bpm,hrv,spo2,fingerDetected,hrvReady,beatQuality
                    timestamp = int(parts[0])
                    ir = int(parts[1])
                    red = int(parts[2])
                    bpm = float(parts[3])
                    hrv = float(parts[4])
                    spo2_raw = float(parts[5])
                    finger_detected = int(parts[6])
                    hrv_ready = int(parts[7])
                    beat_quality = float(parts[8])
                    
                    # Add 85 to SpO2 for display (Arduino sends raw value)
                    spo2 = min(spo2_raw + 85, 100) if spo2_raw > 0 else 0
                    
                    # ML format: time,ir,red,bpm,hrv,spo2,fingerDetected,hrvReady,beatQuality
                    ml_row = f"{timestamp},{ir},{red},{bpm},{hrv},{spo2},{finger_detected},{hrv_ready},{beat_quality}"
                    csv_buffer.append(ml_row)
                    
                    if len(csv_buffer) >= SAVE_CHUNK_SIZE:
                        save_csv_chunk()
                    
                    time_buffer.append(len(time_buffer))
                    ir_buffer.append(ir)
                    red_buffer.append(red)
                    
                    # Smooth BPM (EMA)
                    if bpm > 0:
                        bpm_smooth = bpm_smooth * 0.85 + bpm * 0.15 if bpm_smooth > 0 else bpm
                    
                    # Calculate IBI from BPM if available
                    ibi = (60000.0 / bpm) if bpm > 0 else 0
                    
                    current_vitals.update({
                        'bpm': bpm_smooth if bpm_smooth > 0 else bpm,
                        'spo2': spo2, 'hrv': hrv,
                        'ibi': ibi, 'quality': beat_quality,
                        'finger': bool(finger_detected)
                    })
        except Exception:
            pass
    
    def normalize_ppg(data):
        """DC removal + scale by range so pulse is visible (not flat line)"""
        if len(data) < 2:
            return np.array(data)
        arr = np.array(data, dtype=float)
        dc_removed = arr - np.mean(arr)
        r = np.ptp(dc_removed)
        if r < 1:
            return dc_removed
        return dc_removed / (r / 2)  # Scale to roughly ±1
    
    def animate(frame):
        for _ in range(5):
            read_serial()
        
        if len(time_buffer) > 0:
            line_ir.set_data(list(time_buffer), list(ir_buffer))
            line_red.set_data(list(time_buffer), list(red_buffer))
            ax1.set_xlim(max(0, len(time_buffer) - BUFFER_SIZE), len(time_buffer))
            
            if len(ir_buffer) > 0:
                ir_arr = np.array(ir_buffer)
                red_arr = np.array(red_buffer)
                y_min = min(ir_arr.min(), red_arr.min()) * 0.95
                y_max = max(ir_arr.max(), red_arr.max()) * 1.05
                if y_max > y_min:
                    ax1.set_ylim(y_min, y_max)
            
            if len(ir_buffer) > 10:
                ir_norm = normalize_ppg(ir_buffer)
                red_norm = normalize_ppg(red_buffer)
                t_plot = list(time_buffer)
                line_ir_norm.set_data(t_plot, ir_norm)
                line_red_norm.set_data(t_plot, red_norm)
                ax2.set_xlim(max(0, len(time_buffer) - BUFFER_SIZE), len(time_buffer))
                m = max(np.ptp(ir_norm), np.ptp(red_norm), 0.5)
                ax2.set_ylim(-m * 1.2, m * 1.2)
        
        if current_vitals['finger']:
            vitals_str = (
                f"💓 BPM: {current_vitals['bpm']:5.1f}  |  "
                f"🫁 SpO2: {current_vitals['spo2']:5.1f}%  |  "
                f"📊 HRV: {current_vitals['hrv']:5.1f}ms  |  "
                f"⏱️ IBI: {current_vitals['ibi']:.0f}ms  |  "
                f"Signal: {current_vitals['quality']:.0f}%  |  "
                f"Buffered: {len(csv_buffer)}/{SAVE_CHUNK_SIZE}"
            )
        else:
            vitals_str = f"👆 No finger - Buffered: {len(csv_buffer)}/{SAVE_CHUNK_SIZE}"
        
        vitals_text.set_text(vitals_str)
        
        return line_ir, line_red, line_ir_norm, line_red_norm, vitals_text
    
    ani = animation.FuncAnimation(fig, animate, interval=50, blit=False, cache_frame_data=False)
    
    plt.show()
    
    # Save remaining data on exit
    if len(csv_buffer) > 0:
        save_csv_chunk()

except serial.SerialException as e:
    print(f"❌ Serial error: {e}")
    print("\nOn Raspberry Pi, try: ls /dev/tty* to find port")
    
except KeyboardInterrupt:
    print("\n\n👋 Stopping...")
    if len(csv_buffer) > 0:
        save_csv_chunk()
    
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("✓ Serial closed. Data saved.")
