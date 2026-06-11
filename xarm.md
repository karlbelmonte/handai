#include <Arduino.h>
#include <Servo.h>

const int servoPin = 9;
Servo myServo;

int currentAngle = 90;

int snapToDiscrete(int angle) {
    int allowed[] = {0, 30, 60, 90, 120, 150, 180};  // Add these
    int best = allowed[0];
    int bestDiff = abs(angle - best);

    for (int i = 1; i < 7; i++) {  // Change 4 to 7
        int diff = abs(angle - allowed[i]);
        if (diff < bestDiff) {
            bestDiff = diff;
            best = allowed[i];
        }
    }
    return best;
}

int getSignedValue(const String& input) {
    int colonPos = input.indexOf(':');
    if (colonPos < 0 || colonPos >= input.length() - 1) return 0;
    return input.substring(colonPos + 1).toInt();
}

void setup() {
    Serial.begin(115200);
    Serial.setTimeout(100);
    myServo.attach(servoPin);
    myServo.write(currentAngle);
}

void loop() {
    if (Serial.available() > 0) {
        String input = Serial.readStringUntil('\n');
        input.trim();

        int colonPos = input.indexOf(':');
        if (colonPos <= 0 || colonPos >= input.length() - 1) return;

        char letter = input.charAt(0);
        if (letter != 'A' && letter != 'C' && letter != 'I' && letter != 'X') return;

        int xValue = getSignedValue(input);

        int rawAngle = 90 + (-xValue * 2);
        if (rawAngle < 0) rawAngle = 0;
        if (rawAngle > 180) rawAngle = 180;

        int newAngle = snapToDiscrete(rawAngle);

        if (newAngle != currentAngle) {
            currentAngle = newAngle;
            myServo.write(currentAngle);
        }
    }
}

