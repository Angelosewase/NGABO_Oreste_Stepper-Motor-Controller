#include <Stepper.h>

// Try different step values - test which one works
const int stepsPerRevolution = 2038; // Standard for 28BYJ-48

// Motor pins
const int in1Pin = 8;
const int in2Pin = 9;
const int in3Pin = 10;
const int in4Pin = 11;

Stepper myStepper(stepsPerRevolution, in1Pin, in3Pin, in2Pin, in4Pin);

void setup() {
  pinMode(in1Pin, OUTPUT);
  pinMode(in2Pin, OUTPUT);
  pinMode(in3Pin, OUTPUT);
  pinMode(in4Pin, OUTPUT);
  
  Serial.begin(9600);
  while (!Serial) {
    ; // wait for serial port to connect
  }
  
  Serial.println("Stepper Motor Controller - DEBUG MODE");
  Serial.println("Enter: 'rotate X' where X is degrees");
  Serial.println();
  
  myStepper.setSpeed(5); // Start with slower speed
  
  // Test the motor with a small movement
  Serial.println("Performing test rotation...");
  myStepper.step(100); // Small test movement
  delay(1000);
  myStepper.step(-100); // Return to original position
  Serial.println("Test completed. Ready for commands.");
}

void loop() {
  if (Serial.available() > 0) {
    String input = Serial.readStringUntil('\n');
    input.trim();
    processCommand(input);
  }
}

void processCommand(String command) {
  command.toLowerCase();
  
  if (command.startsWith("rotate")) {
    String degreeStr = command.substring(6);
    degreeStr.trim();
    
    int degrees = degreeStr.toInt();
    
    if (degreeStr.length() == 0) {
      Serial.println("Error: No degree value provided");
      return;
    }
    
    int steps = degreesToSteps(degrees);
    
    Serial.print("Command: ");
    Serial.print(degrees);
    Serial.print("° -> ");
    Serial.print(steps);
    Serial.println(" steps");
    
    Serial.println("Starting rotation...");
    myStepper.step(steps);
    Serial.println("Rotation completed!");
    
  } else {
    Serial.println("Unknown command. Use: rotate X");
  }
}

int degreesToSteps(int degrees) {
  return (degrees * stepsPerRevolution) / 360;
}