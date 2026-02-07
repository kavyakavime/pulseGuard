import serial
import time
import os

PORT = "/dev/cu.usbserial-0001"
BAUD = 115200

print("🫀 PulseGuard - Biosignal Monitor")
print("=" * 50)
print(f"Connecting to {PORT}...")

try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    time.sleep(2)  # Wait for Arduino to initialize
    
    print("✓ Connected! Reading vitals...\n")
    
    # Skip the header line
    header = ser.readline().decode("utf-8", errors="ignore").strip()
    
    while True:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        
        if line and not line.startswith("ERROR"):
            try:
                # Parse CSV: time,ir,red,bpm,hrv,spo2,fingerDetected,hrvReady,beatQuality
                parts = line.split(",")
                
                if len(parts) >= 9:
                    timestamp = int(parts[0])
                    ir = int(parts[1])
                    red = int(parts[2])
                    bpm = float(parts[3])
                    hrv = float(parts[4])
                    spo2_raw = float(parts[5])
                    finger_detected = int(parts[6])
                    hrv_ready = int(parts[7])
                    beat_quality = float(parts[8])
                    
                    # Calibrate SpO2: add 12, clamp between 90-100
                    spo2 = min(max(spo2_raw + 12, 90), 100) if spo2_raw > 0 else 0
                    
                    # Clear screen for live update (optional - comment out if you want scrolling)
                    # os.system('clear' if os.name == 'posix' else 'cls')
                    
                    # Display formatted vitals
                    print(f"\r[{timestamp/1000:.1f}s] ", end="")
                    
                    if finger_detected:
                        print(f"💓 BPM: {bpm:5.1f} | ", end="")
                        print(f"🫁 SpO2: {spo2:5.1f}% | ", end="")
                        
                        if hrv_ready:
                            print(f"📊 HRV: {hrv:5.1f}ms | ", end="")
                        else:
                            print(f"📊 HRV: ------ | ", end="")
                        
                        # Signal quality bar
                        quality_bars = int(beat_quality / 10)
                        quality_str = "█" * quality_bars + "░" * (10 - quality_bars)
                        print(f"Signal: {quality_str} ({beat_quality:.0f}%)", end="")
                    else:
                        print("👆 No finger detected - Place finger on sensor", end="")
                    
                    print("", flush=True)
                else:
                    # Print raw line if not CSV format (errors, debug messages)
                    print(line)
                    
            except (ValueError, IndexError) as e:
                # Print line as-is if parsing fails
                print(line)
        
        elif line.startswith("ERROR"):
            print(f"⚠️  {line}")
            break

except serial.SerialException as e:
    print(f"❌ Serial connection error: {e}")
    print("\nTroubleshooting:")
    print("1. Check if Arduino is connected")
    print("2. Verify PORT is correct (run 'ls /dev/cu.*' to list ports)")
    print("3. Make sure no other program is using the serial port")
    
except KeyboardInterrupt:
    print("\n\n👋 Monitoring stopped by user")
    ser.close()
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    
finally:
    if 'ser' in locals() and ser.is_open:
        ser.close()
        print("Serial connection closed.")
