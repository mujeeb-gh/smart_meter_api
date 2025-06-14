#include <Wire.h>
#include <LiquidCrystal_I2C.h>
#include <driver/adc.h>
#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>

// WiFi credentials - THEY NEED TO UPDATE THESE
const char* ssid = "YOUR_WIFI_SSID";
const char* password = "YOUR_WIFI_PASSWORD";

// API endpoint - UPDATE AFTER DEPLOYMENT
const char* apiURL = "https://your-api-url.railway.app/data";

// Define ADC channels for voltage and current
const adc1_channel_t voltageChannel = ADC1_CHANNEL_6;  // GPIO34
const adc1_channel_t currentChannel = ADC1_CHANNEL_7;  // GPIO35

// Define relay control pins
const int relay1Pin = GPIO_NUM_14;
const int relay2Pin = GPIO_NUM_27;
const int relay3Pin = GPIO_NUM_26;
const int relay4Pin = GPIO_NUM_33;

// LCD configuration
const int LCD_I2C_ADDRESS = 0x27;
const int LCD_COLS = 16;
const int LCD_ROWS = 2;
LiquidCrystal_I2C lcd(LCD_I2C_ADDRESS, LCD_COLS, LCD_ROWS);

// Calibration constants
const float zero_point_ADC_voltage = 2524.576;
const float voltage_calibration_factor = 0.4033;
const float zero_current_ADC_offset = 2543.270;
const float current_calibration_factor = 0.0104;

const int numSamples = 1000;
const int sampleDelayUs = 50;
const adc_atten_t ADC_ATTEN = ADC_ATTEN_DB_11;

// Timing for API calls
unsigned long lastAPICall = 0;
const unsigned long API_INTERVAL = 30000; // Send data every 30 seconds

void setup() {
  Serial.begin(115200);
  Wire.begin(21, 22);
  lcd.begin();
  lcd.backlight();
  lcd.clear();

  // Configure ADC
  adc1_config_width(ADC_WIDTH_BIT_12);
  adc1_config_channel_atten(voltageChannel, ADC_ATTEN);
  adc1_config_channel_atten(currentChannel, ADC_ATTEN);

  // Configure relay pins
  pinMode(relay1Pin, OUTPUT);
  pinMode(relay2Pin, OUTPUT);
  pinMode(relay3Pin, OUTPUT);
  pinMode(relay4Pin, OUTPUT);
  
  // Turn off all relays
  digitalWrite(relay1Pin, HIGH);
  digitalWrite(relay2Pin, HIGH);
  digitalWrite(relay3Pin, HIGH);
  digitalWrite(relay4Pin, HIGH);

  // Connect to WiFi
  connectToWiFi();
}

void loop() {
  // Measure voltage and current (keep existing logic)
  float actual_rms_voltage = measureVoltage();
  float actual_rms_current = measureCurrent();
  float power = actual_rms_voltage * actual_rms_current;

  // Display on LCD (keep existing logic)
  updateLCD(actual_rms_voltage, actual_rms_current, power);

  // Send to API every 30 seconds
  if (millis() - lastAPICall >= API_INTERVAL) {
    sendDataToAPI(actual_rms_voltage, actual_rms_current);
    lastAPICall = millis();
  }

  delay(500);
}

void connectToWiFi() {
  WiFi.begin(ssid, password);
  lcd.setCursor(0, 0);
  lcd.print("Connecting WiFi");
  
  while (WiFi.status() != WL_CONNECTED) {
    delay(1000);
    Serial.print(".");
  }
  
  Serial.println("\nWiFi connected!");
  Serial.print("IP address: ");
  Serial.println(WiFi.localIP());
  
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.print("WiFi Connected");
  delay(2000);
  lcd.clear();
}

void sendDataToAPI(float voltage, float current) {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(apiURL);
    http.addHeader("Content-Type", "application/json");
    
    // Create JSON payload
    StaticJsonDocument<200> doc;
    doc["voltage"] = voltage;
    doc["current"] = current;
    
    String jsonString;
    serializeJson(doc, jsonString);
    
    // Send POST request
    int httpResponseCode = http.POST(jsonString);
    
    if (httpResponseCode > 0) {
      String response = http.getString();
      Serial.println("API Response: " + response);
    } else {
      Serial.print("Error sending data: ");
      Serial.println(httpResponseCode);
    }
    
    http.end();
  } else {
    Serial.println("WiFi not connected");
  }
}

float measureVoltage() {
  long sumOfSquaresVoltage = 0;
  for (int i = 0; i < numSamples; i++) {
    int rawValue = adc1_get_raw(voltageChannel);
    int centeredValue = rawValue - zero_point_ADC_voltage;
    sumOfSquaresVoltage += (long)centeredValue * centeredValue;
    delayMicroseconds(sampleDelayUs);
  }
  
  float rms_analog_voltage = sqrt((float)sumOfSquaresVoltage / numSamples);
  float actual_rms_voltage = rms_analog_voltage * voltage_calibration_factor;
  
  // Apply adjustments
  if (actual_rms_voltage < 10) {
    actual_rms_voltage -= 5.0;
  } else {
    actual_rms_voltage += 4.0;
  }
  
  return actual_rms_voltage;
}

float measureCurrent() {
  long sumOfSquaresCurrent = 0;
  for (int i = 0; i < numSamples; i++) {
    int rawValue = adc1_get_raw(currentChannel);
    int centeredValue = rawValue - zero_current_ADC_offset;
    sumOfSquaresCurrent += (long)centeredValue * centeredValue;
    delayMicroseconds(sampleDelayUs);
  }
  
  float rms_analog_current = sqrt((float)sumOfSquaresCurrent / numSamples);
  float actual_rms_current = rms_analog_current * current_calibration_factor;
  
  // Apply adjustments
  if (actual_rms_current < 0.1) {
    actual_rms_current -= 0.09;
  } else {
    actual_rms_current += 0.2;
  }
  
  return actual_rms_current;
}

void updateLCD(float voltage, float current, float power) {
  lcd.setCursor(0, 0);
  lcd.printf("V:%.2f I:%.2f", voltage, current);
  
  lcd.setCursor(0, 1);
  lcd.printf("P:%.2f W", power);
  
  Serial.print("Voltage: ");
  Serial.print(voltage, 2);
  Serial.print(" V, Current: ");
  Serial.print(current, 2);
  Serial.print(" A, Power: ");
  Serial.print(power, 2);
  Serial.println(" W");
}

void processCommand(String cmd) {
  cmd.toUpperCase();
  if (cmd == "ON1") digitalWrite(relay1Pin, LOW);
  else if (cmd == "OFF1") digitalWrite(relay1Pin, HIGH);
  else if (cmd == "ON2") digitalWrite(relay2Pin, LOW);
  else if (cmd == "OFF2") digitalWrite(relay2Pin, HIGH);
  else if (cmd == "ON3") digitalWrite(relay3Pin, LOW);
  else if (cmd == "OFF3") digitalWrite(relay3Pin, HIGH);
  else if (cmd == "ON4") digitalWrite(relay4Pin, LOW);
  else if (cmd == "OFF4") digitalWrite(relay4Pin, HIGH);
  else Serial.println("Invalid Command. Use ON1-OFF4.");
}