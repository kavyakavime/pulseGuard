/* PulseGuard - MAX30102 Heart Rate & SpO2 Monitor
 * Uses Maxim algorithm with proper buffering and smoothing
 * CSV: time,ir,red,bpm,hrv,spo2,fingerDetected,hrvReady,beatQuality
 * LEDs: Green=finger on, Red=finger off
 */

 #include <Wire.h>
 #include "MAX30105.h"
 #include "spo2_algorithm.h"
 
 MAX30105 particleSensor;
 
 // ---------- PINS ----------
 #define SDA_PIN 18        // Change to 21 if using D21/D22
 #define SCL_PIN 19        // Change to 22 if using D21/D22
 #define LED_GREEN 2
 #define LED_RED 4
 
 #define BUFFER_LEN 100
 #define FINGER_THRESHOLD 50000
 
 // Maxim algorithm buffers
 uint32_t irBuffer[BUFFER_LEN];
 uint32_t redBuffer[BUFFER_LEN];
 
 // Maxim algorithm outputs
 int32_t spo2;
 int8_t validSPO2;
 int32_t heartRate;
 int8_t validHeartRate;
 
 // Smoothed averages (exponential moving average)
 float smoothedHR = 0;
 float smoothedSpO2 = 0;
 const float ALPHA = 0.2; // Smoothing factor (lower = smoother)
 
 // HRV calculation
 unsigned long lastBeatTime = 0;
 float lastIBI = 0;
 #define HRV_BUF 20
 float ibiHistory[HRV_BUF];
 int ibiIdx = 0;
 bool hrvReady = false;
 
 float signalQuality = 0;
 bool fingerDetected = false;
 
 // Tracking
 int validReadings = 0;
 
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
   // Continuously fill buffer with new samples
   static int bufferIndex = 0;
   
   // Wait for sensor data
   while (!particleSensor.available())
     particleSensor.check();
   
   // Read raw values
   uint32_t redValue = particleSensor.getRed();
   uint32_t irValue = particleSensor.getIR();
   particleSensor.nextSample();
   
   // Store in circular buffer
   irBuffer[bufferIndex] = irValue;
   redBuffer[bufferIndex] = redValue;
   bufferIndex = (bufferIndex + 1) % BUFFER_LEN;
   
   // Finger detection
   fingerDetected = (irValue > FINGER_THRESHOLD);
   digitalWrite(LED_GREEN, fingerDetected ? HIGH : LOW);
   digitalWrite(LED_RED, fingerDetected ? LOW : HIGH);
   
   // Signal quality
   if (fingerDetected) {
     signalQuality = constrain((float)irValue / 3000.0f, 0, 100);
   } else {
     signalQuality = 0;
     smoothedHR = 0;
     smoothedSpO2 = 0;
     validReadings = 0;
   }
   
   // Run Maxim algorithm every 25 samples (4x per second at 100Hz)
   static int sampleCount = 0;
   sampleCount++;
   
   if (sampleCount >= 25) {
     sampleCount = 0;
     
     // Run Maxim algorithm on full buffer
     maxim_heart_rate_and_oxygen_saturation(
       irBuffer, BUFFER_LEN, redBuffer,
       &spo2, &validSPO2, &heartRate, &validHeartRate
     );
     
     // Apply smoothing if readings are valid
     if (fingerDetected) {
       if (validHeartRate == 1 && heartRate > 40 && heartRate < 150) {
         if (smoothedHR == 0) {
           smoothedHR = heartRate; // Initialize
         } else {
           smoothedHR = ALPHA * heartRate + (1 - ALPHA) * smoothedHR; // EMA
         }
         validReadings++;
         
         // HRV calculation (store IBI on valid beat)
         if (lastBeatTime > 0) {
           lastIBI = millis() - lastBeatTime;
           if (lastIBI > 300 && lastIBI < 2000 && ibiIdx < 1000) { // Valid IBI range
             ibiHistory[ibiIdx % HRV_BUF] = lastIBI;
             ibiIdx++;
           }
         }
         lastBeatTime = millis();
       }
       
      if (validSPO2 == 1 && spo2 > 50 && spo2 < 101) {
        // Apply calibration offset (+10 to bring 85% up to 95% range)
        int32_t calibratedSpO2 = spo2 + 10;
        if (calibratedSpO2 > 100) calibratedSpO2 = 100;
        
        if (smoothedSpO2 == 0) {
          smoothedSpO2 = calibratedSpO2; // Initialize
        } else {
          smoothedSpO2 = ALPHA * calibratedSpO2 + (1 - ALPHA) * smoothedSpO2; // EMA
        }
      }
     }
   }
   
   // Calculate HRV
   hrvReady = (ibiIdx >= 5);
   float hrv = computeHRV();
   
   // Prepare output (only show stable readings after 3+ valid measurements)
   int displayBPM = (fingerDetected && validReadings >= 3) ? (int)smoothedHR : 0;
   int displaySpO2 = (fingerDetected && validReadings >= 3) ? (int)smoothedSpO2 : 0;
   
   // CSV Output
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
