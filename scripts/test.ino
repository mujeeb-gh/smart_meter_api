#include <Wire.h>  // Library for I2C communication
#include <LiquidCrystal_I2C.h>  // LCD over I2C
#include <driver/adc.h>  // For low-level ADC reading

// Define ADC channels for voltage and current
const adc1_channel_t voltageChannel = ADC1_CHANNEL_6;  // GPIO34
const adc1_channel_t currentChannel = ADC1_CHANNEL_7;  // GPIO35

// Define relay control pins
const int relay1Pin = GPIO_NUM_14;
const int relay2Pin = GPIO_NUM_27;
const int relay3Pin = GPIO_NUM_26;
const int relay4Pin = GPIO_NUM_33;

// LCD configuration (I2C address, columns, rows)
const int LCD_I2C_ADDRESS = 0x27;
const int LCD_COLS = 16;
const int LCD_ROWS = 2;
LiquidCrystal_I2C lcd(LCD_I2C_ADDRESS, LCD_COLS, LCD_ROWS);

// Calibration constants for voltage
const float zero_point_ADC_voltage = 2524.576;  // Midpoint of ADC readings when no voltage
const float voltage_calibration_factor = 0.4033;  // To convert ADC to real volts

// Calibration constants for current
const float zero_current_ADC_offset = 2543.270;  // Midpoint of ADC readings when no current
const float current_calibration_factor = 0.0104;  // To convert ADC to real amps

const int numSamples = 1000;  // Number of samples to take for RMS calculation
const int sampleDelayUs = 50;  // Delay between samples in microseconds

const adc_atten_t ADC_ATTEN = ADC_ATTEN_DB_11;  // Set attenuation for full 0–3.3V range

void setup() {
  Serial.begin(115200);  // Start serial communication for debugging
  Wire.begin(21, 22);  // Start I2C on GPIO21 (SDA), GPIO22 (SCL)
  lcd.begin();         // Initialize LCD
  lcd.backlight();     // Turn on LCD backlight
  lcd.clear();         // Clear LCD screen

  // Configure ADC settings
  adc1_config_width(ADC_WIDTH_BIT_12);  // 12-bit resolution
  adc1_config_channel_atten(voltageChannel, ADC_ATTEN);
  adc1_config_channel_atten(currentChannel, ADC_ATTEN);

  // Set all relay pins as output
  pinMode(relay1Pin, OUTPUT);
  pinMode(relay2Pin, OUTPUT);
  pinMode(relay3Pin, OUTPUT);
  pinMode(relay4Pin, OUTPUT);

  // Turn off all relays initially (HIGH = off for active-low relays)
  digitalWrite(relay1Pin, HIGH);
  digitalWrite(relay2Pin, HIGH);
  digitalWrite(relay3Pin, HIGH);
  digitalWrite(relay4Pin, HIGH);
}

void loop() {
  // --- Measure RMS voltage ---
  long sumOfSquaresVoltage = 0;
  for (int i = 0; i < numSamples; i++) {
    int rawValue = adc1_get_raw(voltageChannel);  // Get raw ADC value
    int centeredValue = rawValue - zero_point_ADC_voltage;  // Remove DC offset
    sumOfSquaresVoltage += (long)centeredValue * centeredValue;  // Square and accumulate
    delayMicroseconds(sampleDelayUs);  // Small delay
  }
  // Calculate RMS analog value and convert to actual voltage
  float rms_analog_voltage = sqrt((float)sumOfSquaresVoltage / numSamples);
  float actual_rms_voltage = rms_analog_voltage * voltage_calibration_factor;

  // --- Measure RMS current ---
  long sumOfSquaresCurrent = 0;
  for (int i = 0; i < numSamples; i++) {
    int rawValue = adc1_get_raw(currentChannel);  // Get raw ADC value
    int centeredValue = rawValue - zero_current_ADC_offset;  // Remove DC offset
    sumOfSquaresCurrent += (long)centeredValue * centeredValue;  // Square and accumulate
    delayMicroseconds(sampleDelayUs);
  }
  // Calculate RMS analog value and convert to actual current
  float rms_analog_current = sqrt((float)sumOfSquaresCurrent / numSamples);
  float actual_rms_current = rms_analog_current * current_calibration_factor;

  // --- Adjust based on whether plug is in socket ---
  if (actual_rms_voltage < 10) {
    // No socket connection – adjust to zero
    actual_rms_voltage -= 5.0;
    actual_rms_current -= 0.09;
  } else {
    // Socket connected – apply positive offset to correct
    actual_rms_voltage += 4.0;
    actual_rms_current += 0.2;
  }

  // --- Calculate power (P = V x I) ---
  float power = actual_rms_voltage * actual_rms_current;

  // --- Show values on LCD ---
  lcd.setCursor(0, 0);  // First row
  lcd.printf("V:%.2f I:%.2f", actual_rms_voltage, actual_rms_current);  // Compact display

  lcd.setCursor(0, 1);  // Second row
  lcd.printf("P:%.2f W", power);  // Show power

  // --- Debug output to Serial Monitor ---
  Serial.print("Voltage: ");
  Serial.print(actual_rms_voltage, 2);
  Serial.print(" V, Current: ");
  Serial.print(actual_rms_current, 2);
  Serial.print(" A, Power: ");
  Serial.print(power, 2);
  Serial.println(" W");

  delay(500);  // Wait before repeating
}

// --- Function to control relays based on command string (e.g., "ON1", "OFF2") ---
void processCommand(String cmd) {
  cmd.toUpperCase();  // Convert command to uppercase

  if (cmd == "ON1") digitalWrite(relay1Pin, LOW);
  else if (cmd == "OFF1") digitalWrite(relay1Pin, HIGH);
  else if (cmd == "ON2") digitalWrite(relay2Pin, LOW);
  else if (cmd == "OFF2") digitalWrite(relay2Pin, HIGH);
  else if (cmd == "ON3") digitalWrite(relay3Pin, LOW);
  else if (cmd == "OFF3") digitalWrite(relay3Pin, HIGH);
  else if (cmd == "ON4") digitalWrite(relay4Pin, LOW);
  else if (cmd == "OFF4") digitalWrite(relay4Pin, HIGH);
  else Serial.println("Invalid Command. Use ON1-OFF4.");  // Error message
}
