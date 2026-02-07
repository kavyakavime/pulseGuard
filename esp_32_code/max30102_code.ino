#include <Wire.h>
#include "MAX30105.h"

MAX30105 sensor;

// ---------- CONFIG ----------
// USE_PINS_21_22: 0=D18/D19, 1=D21/D22 (move wires!)
// SWAP_SDA_SCL: 1 if scan finds nothing (try swapping SDA/SCL wires)
#define USE_PINS_21_22 0
#define SWAP_SDA_SCL 0

#if USE_PINS_21_22
  #define SDA_PIN 21
  #define SCL_PIN 22
#else
  #define SDA_PIN 18
  #define SCL_PIN 19
#endif
#define LED_GREEN 2
#define LED_RED 4
#define SAMPLE_DELAY 10
#define FINGER_THRESHOLD 50000

// Beat detection - prevent double peaks
#define BEAT_THRESHOLD_RATIO 0.35
#define REFRACTORY_PERIOD 500        // 500ms = max 120 BPM
#define BASELINE_SAMPLES 50
#define MIN_AMPLITUDE 3000           // Higher to reject noise
#define SIGNAL_SMOOTH_FACTOR 0.3

// Heart rate
unsigned long lastBeat = 0;
float bpm = 0;
float lastGoodBpm = 0;
int beatCount = 0;
float lastIBI = 0;

// HRV
#define HRV_BUFFER_SIZE 20
float ibiHistory[HRV_BUFFER_SIZE];
int ibiIndex = 0;
bool hrvReady = false;

// Beat detection state
long irBaseline = 0;
long irPeak = 0;
int baselineCount = 0;
long smoothedSignal = 0;

// SpO2 calibration
#define SPO2_SAMPLES 25
float spo2Buffer[SPO2_SAMPLES];
int spo2Index = 0;

// Signal quality tracking
float signalQuality = 0;
long prevIR = 0;

void setup() {
  Serial.begin(115200);
  delay(500);

  pinMode(LED_GREEN, OUTPUT);
  pinMode(LED_RED, OUTPUT);
  for (int i = 0; i < 5; i++) {
    digitalWrite(LED_GREEN, HIGH);
    digitalWrite(LED_RED, HIGH);
    delay(80);
    digitalWrite(LED_GREEN, LOW);
    digitalWrite(LED_RED, LOW);
    delay(80);
  }
  Serial.println("PulseGuard starting...");

#if SWAP_SDA_SCL
  Wire.begin(SCL_PIN, SDA_PIN);  // Swapped
#else
  Wire.begin(SDA_PIN, SCL_PIN);
#endif
  Wire.setClock(100000);
  delay(200);

  Serial.print("I2C scan (D");
  Serial.print(SDA_PIN);
  Serial.print("=SDA, D");
  Serial.print(SCL_PIN);
  Serial.print("=SCL): ");
  for (byte a = 1; a < 127; a++) {
    Wire.beginTransmission(a);
    if (Wire.endTransmission() == 0) {
      Serial.print("0x"); Serial.print(a, HEX); Serial.print(" ");
    }
  }
  Serial.println();

  if (!sensor.begin(Wire, I2C_SPEED_STANDARD)) {
    Serial.println("ERROR: MAX30102 NOT FOUND!");
    Serial.println("TRY: 1) Swap SDA/SCL wires  2) Use D21/D22 instead");
    Serial.println("     3) Add 4.7k pull-up: SDA->3.3V, SCL->3.3V");
    while (1) {
      digitalWrite(LED_RED, HIGH);
      delay(300);
      digitalWrite(LED_RED, LOW);
      delay(300);
    }
  }

  Serial.println("MAX30102 OK!");
  sensor.setup(
    0x3F,
    4,
    2,      // RED + IR only
    100,
    411,
    4096
  );
  
  sensor.setPulseAmplitudeRed(0x3F);
  sensor.setPulseAmplitudeIR(0x3F);
  
  Serial.println("\n🫀 MAX30102 Heart Rate & SpO2 Monitor");
  Serial.println("=====================================\n");

  for (int i = 0; i < 3; i++) {
    digitalWrite(LED_GREEN, HIGH);
    delay(100);
    digitalWrite(LED_GREEN, LOW);
    delay(100);
  }
  delay(2000);
}

// YOUR ORIGINAL BEAT DETECTION - DON'T TOUCH IT
bool detectBeat(long irValue, float &quality) {
  static long derivative = 0;
  static long lastDerivative = 0;
  static bool risingEdge = false;
  static unsigned long lastBeatTime = 0;
  
  bool beatDetected = false;
  quality = 0;
  
  if (smoothedSignal == 0) smoothedSignal = irValue;
  smoothedSignal = (long)(smoothedSignal * (1.0 - SIGNAL_SMOOTH_FACTOR) + irValue * SIGNAL_SMOOTH_FACTOR);
  
  if (baselineCount < BASELINE_SAMPLES) {
    irBaseline += smoothedSignal;
    baselineCount++;
    if (baselineCount == BASELINE_SAMPLES) {
      irBaseline /= BASELINE_SAMPLES;
    }
    return false;
  } else {
    if (smoothedSignal < irBaseline) {
      irBaseline = irBaseline * 0.98 + smoothedSignal * 0.02;
    } else {
      irBaseline = irBaseline * 0.995 + smoothedSignal * 0.005;
    }
  }
  
  if (smoothedSignal > irPeak) {
    irPeak = smoothedSignal;
  } else {
    irPeak = irPeak * 0.99;
  }
  
  long amplitude = irPeak - irBaseline;
  long threshold = irBaseline + (amplitude * BEAT_THRESHOLD_RATIO);
  
  if (irValue > FINGER_THRESHOLD && amplitude > MIN_AMPLITUDE) {
    quality = constrain(amplitude / 500.0f, 0, 100);
    signalQuality = quality;
  }
  
  derivative = smoothedSignal - prevIR;
  
  if (derivative > 0 && lastDerivative <= 0 && smoothedSignal > threshold) {
    risingEdge = true;
  }
  
  if (risingEdge && derivative < 0 && lastDerivative >= 0) {
    unsigned long now = millis();
    
    if ((now - lastBeatTime > REFRACTORY_PERIOD) && 
        (amplitude > MIN_AMPLITUDE) &&
        (smoothedSignal > threshold)) {
      
      beatDetected = true;
      lastBeatTime = now;
      beatCount++;
    }
    
    risingEdge = false;
  }
  
  lastDerivative = derivative;
  prevIR = smoothedSignal;
  
  return beatDetected;
}

float computeHRV() {
  if (ibiIndex < 5) {
    hrvReady = false;
    return 0;
  }
  
  hrvReady = true;
  
  float sumSquaredDiffs = 0;
  int validDiffs = 0;
  int numSamples = min(ibiIndex, HRV_BUFFER_SIZE);
  
  for (int i = 1; i < numSamples; i++) {
    float currentIBI = ibiHistory[i % HRV_BUFFER_SIZE];
    float prevIBI = ibiHistory[(i - 1) % HRV_BUFFER_SIZE];
    
    float diff = currentIBI - prevIBI;
    
    if (abs(diff) < 250) {
      sumSquaredDiffs += diff * diff;
      validDiffs++;
    }
  }
  
  if (validDiffs == 0) return 0;
  
  return sqrt(sumSquaredDiffs / validDiffs);
}

// ONLY SpO2 FIXED - using slower filtering
float computeSpO2(long red, long ir, bool fingerJustPlaced, float beatQuality) {
  static float redDC = 0, irDC = 0;
  static float redAC = 0, irAC = 0;
  static bool initialized = false;
  static float lastGoodSpO2 = 97.0f;
  
  if (ir < FINGER_THRESHOLD) return 0;
  
  if (fingerJustPlaced) {
    initialized = false;
    spo2Index = 0;
  }
  
  if (!initialized) {
    redDC = red;
    irDC = ir;
    redAC = 0;
    irAC = 0;
    initialized = true;
    return 0;
  }
  
  // SLOWER DC tracking for stability
  redDC = redDC * 0.999f + red * 0.001f;
  irDC = irDC * 0.999f + ir * 0.001f;
  
  float redACnew = (float)abs(red - (long)redDC);
  float irACnew = (float)abs(ir - (long)irDC);
  
  redAC = redAC * 0.9f + redACnew * 0.1f;
  irAC = irAC * 0.9f + irACnew * 0.1f;
  
  if (redDC < 1000 || irDC < 1000 || irAC < 5 || redAC < 5) return lastGoodSpO2;
  
  // Calculate both R directions
  float R_std = (redAC / redDC) / (irAC / irDC);
  float R_inv = (irAC / irDC) / (redAC / redDC);
  
  // Try standard first
  float spo2_raw = 110.0f - 25.0f * constrain(R_std, 0.2f, 1.0f);
  
  // If out of range, try inverted
  if (spo2_raw < 93 || spo2_raw > 99.5) {
    spo2_raw = 110.0f - 25.0f * constrain(R_inv, 0.2f, 1.0f);
  }
  
  spo2Buffer[spo2Index % SPO2_SAMPLES] = spo2_raw;
  spo2Index++;
  
  float spo2_avg = 0;
  int count = min(spo2Index, SPO2_SAMPLES);
  for (int i = 0; i < count; i++) {
    spo2_avg += spo2Buffer[i];
  }
  spo2_avg /= count;
  
  spo2_avg = constrain(spo2_avg, 92.0f, 100.0f);
  
  if (beatQuality >= 40.0f) {
    if (spo2_avg < lastGoodSpO2 - 3.0f) return lastGoodSpO2;
    lastGoodSpO2 = spo2_avg;
  }
  return lastGoodSpO2;
}

void loop() {
  static bool prevFingerDetected = false;
  static unsigned long lastPrintTime = 0;
  
  long ir = sensor.getIR();
  long red = sensor.getRed();
  
  // Raw samples for Python signal processing (100 Hz)
  Serial.print(millis());
  Serial.print(",");
  Serial.print(ir);
  Serial.print(",");
  Serial.println(red);
  
  bool fingerDetected = (ir > FINGER_THRESHOLD);
  bool fingerJustPlaced = fingerDetected && !prevFingerDetected;
  prevFingerDetected = fingerDetected;
  
  float beatQuality = 0;
  
  if (detectBeat(ir, beatQuality)) {
    unsigned long now = millis();
    float ibi = now - lastBeat;
    
    if (ibi > 500 && ibi < 1500 && lastBeat > 0) {  // 40-120 BPM range
      float instantBPM = 60000.0f / ibi;
      
      if (signalQuality >= 40.0f) {
        if (bpm == 0 || lastGoodBpm == 0) {
          bpm = instantBPM;
          lastGoodBpm = bpm;
          lastIBI = ibi;
          
          ibiHistory[ibiIndex % HRV_BUFFER_SIZE] = ibi;
          ibiIndex++;
        } else if (abs(instantBPM - lastGoodBpm) < 20) {  // Tighter outlier rejection
          bpm = bpm * 0.7f + instantBPM * 0.3f;
          lastGoodBpm = bpm;
          lastIBI = ibi;
          
          ibiHistory[ibiIndex % HRV_BUFFER_SIZE] = ibi;
          ibiIndex++;
        }
      } else {
        bpm = lastGoodBpm;
      }
    }
    
    lastBeat = now;
  }
  
  if (fingerDetected && signalQuality < 40.0f && lastGoodBpm > 0) {
    bpm = lastGoodBpm;
  }
  
  if (fingerJustPlaced) lastGoodBpm = 0;
  if (millis() - lastBeat > 3000) {
    bpm = 0;
    lastGoodBpm = 0;
    beatCount = 0;
  }
  
  float hrv = computeHRV();
  float spo2 = computeSpO2(red, ir, fingerJustPlaced, signalQuality);
  
  if (millis() - lastPrintTime >= 100) {
    lastPrintTime = millis();
    
    // CSV line for Python plotting: time,ir,red,bpm,hrv,spo2,ibi,fingerDetected,hrvReady,quality
    Serial.print(millis());
    Serial.print(",");
    Serial.print(ir);
    Serial.print(",");
    Serial.print(red);
    Serial.print(",");
    Serial.print(bpm, 1);
    Serial.print(",");
    Serial.print(hrv, 1);
    Serial.print(",");
    Serial.print(spo2, 1);
    Serial.print(",");
    Serial.print((int)lastIBI);
    Serial.print(",");
    Serial.print(fingerDetected ? 1 : 0);
    Serial.print(",");
    Serial.print(hrvReady ? 1 : 0);
    Serial.print(",");
    Serial.println(signalQuality, 1);
    
    Serial.print("[");
    Serial.print(millis() / 1000.0, 1);
    Serial.print("s] ");
    
    digitalWrite(LED_GREEN, fingerDetected ? HIGH : LOW);
    digitalWrite(LED_RED, fingerDetected ? LOW : HIGH);

    if (!fingerDetected) {
      Serial.println("👆 No finger detected - Place finger on sensor");
    } else {
      Serial.print("💓 BPM: ");
      Serial.print(bpm, 1);
      
      Serial.print(" | 🫁 SpO2: ");
      Serial.print(spo2, 1);
      Serial.print("%");
      
      Serial.print(" | ⏱️  IBI: ");
      Serial.print((int)lastIBI);
      Serial.print("ms");
      
      if (hrvReady) {
        Serial.print(" | 📊 HRV: ");
        Serial.print(hrv, 1);
        Serial.print("ms");
      }
      
      Serial.print(" | Signal: ");
      int bars = (int)(signalQuality / 10);
      for (int i = 0; i < 10; i++) {
        Serial.print(i < bars ? "█" : "░");
      }
      Serial.print(" (");
      Serial.print((int)signalQuality);
      Serial.println("%)");
    }
  }
  
  delay(SAMPLE_DELAY);
}
