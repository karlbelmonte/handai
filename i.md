#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

Adafruit_PWMServoDriver pwm = Adafruit_PWMServoDriver();

const int NUM_SERVOS = 6;
const int servoChannel[NUM_SERVOS] = {0, 1, 2, 3, 4, 5};
int servoAngle[NUM_SERVOS] = {90, 90, 90, 90, 90, 90};

// Function to snap angle to nearest discrete value (0, 30, 60, 90)
int snapToDiscrete(int angle) {
    int discreteAngles[] = {0, 30, 60, 90};
    int closest = discreteAngles[0];
    int minDiff = abs(angle - closest);
    
    for (int i = 1; i < 4; i++) {
        int diff = abs(angle - discreteAngles[i]);
        if (diff < minDiff) {
            minDiff = diff;
            closest = discreteAngles[i];
        }
    }
    return closest;
}

void setup() {
    Serial.begin(115200);
    pwm.begin();
    pwm.setPWMFreq(60);
    Serial.println("6 Servos Ready (PCA9685)");
    Serial.println("Letters: A-F for RIGHT (positive X), G-L for LEFT (negative X)");
    delay(100);
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();
        
        Serial.print("Received: ");
        Serial.println(input);
        
        if (input.length() > 2) {
            char letter = input.charAt(0);
            int value = input.substring(2).toInt();
            
            int servoNum = -1;
            
            // A-F for RIGHT hand (positive X), G-L for LEFT hand (negative X)
            if (letter >= 'A' && letter <= 'F') {
                servoNum = letter - 'A';
            } else if (letter >= 'G' && letter <= 'L') {
                servoNum = letter - 'G';
            }
            
            if (servoNum >= 0 && servoNum < NUM_SERVOS) {
                int newAngle;
                
                if (letter == 'F' || letter == 'L') {
                    // F and L are on/off servos (0 or 90)
                    newAngle = (value == 1) ? 90 : 0;
                } else {
                    // Value 0-90 maps directly to 0-90 degrees
                    // Negative values become 0, values >90 become 90
                    newAngle = constrain(value, 0, 90);
                    // Snap to nearest discrete angle (0, 30, 60, 90)
                    newAngle = snapToDiscrete(newAngle);
                }
                
                if (newAngle != servoAngle[servoNum]) {
                    servoAngle[servoNum] = newAngle;
                    uint16_t pulse = map(newAngle, 0, 180, 150, 600);
                    pwm.setPWM(servoChannel[servoNum], 0, pulse);
                    
                    Serial.print(">>> ");
                    Serial.print(letter);
                    Serial.print("=");
                    Serial.print(value);
                    Serial.print(" -> Servo ");
                    Serial.print(servoNum);
                    Serial.print(": ");
                    Serial.print(newAngle);
                    Serial.println(" deg");
                }
            }
        }
    }
}