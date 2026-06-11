#include <AccelStepper.h>

#define STEP_PIN 4
#define DIR_PIN 3

AccelStepper stepper(AccelStepper::DRIVER, STEP_PIN, DIR_PIN);

void setup() {
    Serial.begin(115200);
    stepper.setMaxSpeed(1000);    // Max speed in steps per second
    stepper.setAcceleration(500); // Acceleration in steps/sec²
}

void loop() {
    stepper.moveTo(200);  // Move 200 steps (1 full rotation for NEMA-17)
    while (stepper.distanceToGo() != 0) {
        stepper.run();
    }
    delay(1000);  // Wait 1 second
    
    stepper.moveTo(-200);  // Move back 200 steps
    while (stepper.distanceToGo() != 0) {
        stepper.run();
    }
    delay(1000);  // Wait 1 second
}