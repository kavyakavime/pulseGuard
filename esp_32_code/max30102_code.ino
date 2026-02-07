/* PulseGuard - MAX30102 Heart Rate & SpO2 Monitor
 * Uses bandpass filtering + peak detection + ratio-of-ratios SpO2
 * CSV: time,ir,red,bpm,hrv,spo2,fingerDetected,hrvReady,beatQuality
 * LEDs: Green=finger on, Red=finger off
 */

#include <Wire.h>
#include "MAX30105.h"

MAX30105 particleSensor;

// ---------- PINS ----------
#define SDA_PIN 18        // Change to 21 if using D21/D22
#define SCL_PIN 19        // Change to 22 if using D21/D22
#define LED_GREEN 2
#define LED_RED 4

#define BUFFER_LEN 100
#define REPORTING_PERIOD_MS 1000
#define FINGER_THRESHOLD 50000

// Bandpass filter arrays (5-point)
float firxv[5] = {0};  // IR input filter values
float firyv[5] = {0};  // IR output filter values
float fredxv[5] = {0}; // Red input filter values
float fredyv[5] = {0}; // Red output filter values

// Heart rate and SpO2 averaging
float hrArray[5] = {0};
float spo2Array[5] = {0};
int arrayIdx = 0;

float heartRate = 0.0;
float spo2 = 0.0;
float lastMeasTime = 0.0;
unsigned long tcnt = 0;

// HRV calculation
float lastIBI = 0;
#define HRV_BUF 20
float ibiHistory[HRV_BUF];
int ibiIdx = 0;
bool hrvReady = false;

float signalQuality = 0;
bool fingerDetected = false;

void setup() {
  Serial.begin(115200);
  delay(500);
  
  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  digitalWrite(LED_GREEN, LOW);
  digitalWrite(LED_RED, HIGH);
  
  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  delay(100);
  
  Serial.println("Initializing Pulse Oximeter..");
  
  if (!particleSensor.begin(Wire, I2C_SPEED_FAST)) {
    Serial.println("MAX30102 NOT FOUND - Check SDA/SCL (try D21/D22 or D18/D19)");
    while (1) {
      digitalWrite(LED_RED, HIGH);
      delay(500);
      digitalWrite(LED_RED, LOW);
      delay(500);
    }
  }
  
  byte ledBrightness = 50;
  byte sampleAverage = 1;
  byte ledMode = 2;
  byte sampleRate = 100;
  int pulseWidth = 69;
  int adcRange = 4096;
  
  particleSensor.setup(ledBrightness, sampleAverage, ledMode, sampleRate, pulseWidth, adcRange);
  
  Serial.println("time,ir,red,bpm,hrv,spo2,fingerDetected,hrvReady,beatQuality");
  Serial.println("PulseGuard ready.");
}

float computeHRV() {
  if (ibiIdx < 5) return 0;
  
  float sum = 0;
  int n = 0;
  
  for (int i = 1; i < min(ibiIdx, HRV_BUF); i++) {
    float d = ibiHistory[i % HRV_BUF] - ibiHistory[(i - 1) % HRV_BUF];
    if (fabs(d) < 250) {
      sum += d * d;
      n++;
    }
  }
  
  return (n > 0) ? sqrt(sum / n) : 0;
}

void loop() {
  // Wait for sensor data
  while (!particleSensor.available())
    particleSensor.check();
  
  // Read raw values
  uint32_t redValue = particleSensor.getRed();
  uint32_t irValue = particleSensor.getIR();
  particleSensor.nextSample();
  
  // Finger detection
  fingerDetected = (irValue > FINGER_THRESHOLD);
  digitalWrite(LED_GREEN, fingerDetected ? HIGH : LOW);
  digitalWrite(LED_RED, fingerDetected ? LOW : HIGH);
  
  // Signal quality
  if (fingerDetected) {
    signalQuality = constrain((float)irValue / 3000.0f, 0, 100);
  } else {
    signalQuality = 0;
  }
  
  // Update measurement time (100 Hz sample rate = 10ms per sample)
  float measTime = 0.01 * tcnt++;
  
  // Apply bandpass filter to IR signal
  firxv[0] = firxv[1];
  firxv[1] = firxv[2];
  firxv[2] = firxv[3];
  firxv[3] = firxv[4];
  firxv[4] = irValue / 3.48311;
  
  firyv[0] = firyv[1];
  firyv[1] = firyv[2];
  firyv[2] = firyv[3];
  firyv[3] = firyv[4];
  firyv[4] = (firxv[0] + firxv[4]) - 2 * firxv[2]
             + (-0.1718123813 * firyv[0]) + (0.3686645260 * firyv[1])
             + (-1.1718123813 * firyv[2]) + (1.9738037992 * firyv[3]);
  
  // Apply bandpass filter to Red signal
  fredxv[0] = fredxv[1];
  fredxv[1] = fredxv[2];
  fredxv[2] = fredxv[3];
  fredxv[3] = fredxv[4];
  fredxv[4] = redValue / 3.48311;
  
  fredyv[0] = fredyv[1];
  fredyv[1] = fredyv[2];
  fredyv[2] = fredyv[3];
  fredyv[3] = fredyv[4];
  fredyv[4] = (fredxv[0] + fredxv[4]) - 2 * fredxv[2]
              + (-0.1718123813 * fredyv[0]) + (0.3686645260 * fredyv[1])
              + (-1.1718123813 * fredyv[2]) + (1.9738037992 * fredyv[3]);
  
  // Peak detection: check if current point is a local maximum
  if (-1.0 * firyv[4] >= 100 && 
      -1.0 * firyv[2] > -1.0 * firyv[0] && 
      -1.0 * firyv[2] > -1.0 * firyv[4]) {
    
    float timeDelta = measTime - lastMeasTime;
    
    // Reject peaks that are too close (< 0.5s = 120+ BPM max)
    if (timeDelta >= 0.5) {
      // Calculate instantaneous heart rate
      float instantHR = 60.0 / timeDelta;
      
      // Calculate instantaneous SpO2 using ratio-of-ratios
      float ratioRed = fredyv[4] / fredxv[4];
      float ratioIR = firyv[4] / firxv[4];
      float instantSpO2 = 110.0 - 25.0 * (ratioRed / ratioIR);
      if (instantSpO2 > 100.0) instantSpO2 = 99.9;
      
      // Store in arrays for averaging
      hrArray[arrayIdx % 5] = instantHR;
      spo2Array[arrayIdx % 5] = instantSpO2;
      arrayIdx++;
      
      // Store IBI for HRV
      lastIBI = timeDelta * 1000.0; // Convert to ms
      if (ibiIdx < 1000) {
        ibiHistory[ibiIdx % HRV_BUF] = lastIBI;
        ibiIdx++;
      }
      
      // Calculate averaged values
      heartRate = (hrArray[0] + hrArray[1] + hrArray[2] + hrArray[3] + hrArray[4]) / 5.0;
      spo2 = (spo2Array[0] + spo2Array[1] + spo2Array[2] + spo2Array[3] + spo2Array[4]) / 5.0;
      
      // Validate ranges
      if (heartRate < 40 || heartRate > 150) heartRate = 0;
      if (spo2 < 50 || spo2 > 101) spo2 = 0;
      
      lastMeasTime = measTime;
    }
  }
  
  // Reset if no beat detected for 1.8 seconds
  if (heartRate > 0 && (measTime - lastMeasTime) > 1.8) {
    fingerDetected = false;
  }
  
  // Calculate HRV
  hrvReady = (ibiIdx >= 5);
  float hrv = computeHRV();
  
  // Output CSV
  int displayBPM = fingerDetected ? (int)heartRate : 0;
  int displaySpO2 = fingerDetected ? (int)spo2 : 0;
  
  Serial.print(millis());
  Serial.print(",");
  Serial.print(irValue);
  Serial.print(",");
  Serial.print(redValue);
  Serial.print(",");
  Serial.print(displayBPM);
  Serial.print(",");
  Serial.print(hrv, 2);
  Serial.print(",");
  Serial.print(displaySpO2);
  Serial.print(",");
  Serial.print(fingerDetected ? 1 : 0);
  Serial.print(",");
  Serial.print(hrvReady ? 1 : 0);
  Serial.print(",");
  Serial.println(signalQuality, 2);
}
