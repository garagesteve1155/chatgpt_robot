#include <Servo.h>

Servo ESC1;
Servo ESC2;
#define TRIGGER_PIN 12
#define ECHO_PIN 13
#define MAX_DISTANCE 200
int right_dir = 4;
int left_dir = 7;
int right_speed = 5;
int left_speed = 6;
int center_pos = 90;
int left_pos = 170;
int right_pos = 10;
void setup() {
  pinMode(TRIGGER_PIN, OUTPUT);
  pinMode(ECHO_PIN, INPUT);

  pinMode(right_dir, OUTPUT);
  pinMode(right_speed, OUTPUT);
  pinMode(left_dir, OUTPUT);
  pinMode(left_speed, OUTPUT);
  
  digitalWrite(right_dir, LOW);
  digitalWrite(left_dir, LOW);
  digitalWrite(right_speed, LOW);
  digitalWrite(left_speed, LOW);
  Serial.begin(9600);
  ESC1.attach(9);
  ESC2.attach(10);
  ESC1.write(90);
  ESC2.write(center_pos);
}

void loop() {
  char val = Serial.read();
  processInput(val);
  delay(10);
}

unsigned int measureDistance() {
  digitalWrite(TRIGGER_PIN, LOW);
  delayMicroseconds(2);
  digitalWrite(TRIGGER_PIN, HIGH);
  delayMicroseconds(10);
  digitalWrite(TRIGGER_PIN, LOW);
  unsigned long duration = pulseIn(ECHO_PIN, HIGH);
  unsigned int distance = duration * 0.034 / 2;
  if (distance > MAX_DISTANCE) {
    distance = MAX_DISTANCE;
  }
  return distance;
}

void sweepServo(Servo &servo, int targetPos, int speedDelay) {
  int currentPos = servo.read();
  if (currentPos < targetPos) {
    for (int pos = currentPos; pos <= targetPos; pos++) {
      servo.write(pos);
      delay(speedDelay);
    }
  } else {
    for (int pos = currentPos; pos >= targetPos; pos--) {
      servo.write(pos);
      delay(speedDelay);
    }
  }
}

void processInput(char val) {
  switch(val) {

    case 'w':
      analogWrite(right_speed, 150);
      analogWrite(left_speed, 150);
      digitalWrite(right_dir, LOW);
      digitalWrite(left_dir, LOW);
      break;
    case 's':
      analogWrite(right_speed, 150);
      analogWrite(left_speed, 150);
      digitalWrite(right_dir, HIGH);
      digitalWrite(left_dir, HIGH);
      break;
    case 'a':
      analogWrite(right_speed, 200);
      analogWrite(left_speed, 200);
      digitalWrite(right_dir, HIGH);
      digitalWrite(left_dir, LOW);
      break;
    case 'd':
      analogWrite(right_speed, 200);
      analogWrite(left_speed, 200);
      digitalWrite(right_dir, LOW);
      digitalWrite(left_dir, HIGH);
      break;
    case 'x':
      analogWrite(right_speed, LOW);
      analogWrite(left_speed, LOW);
      break;
    case '1':
      ESC1.attach(9);
      sweepServo(ESC1, 90, 6);
      ESC1.detach();
      break;
    case '2':
      ESC1.attach(9);
      sweepServo(ESC1, 140, 9);
      break;
    case '3':
      ESC2.attach(10);
      sweepServo(ESC2, left_pos, 9);
      ESC2.detach();
      break;
    case '4':
      ESC2.attach(10);
      sweepServo(ESC2, center_pos, 9);
      ESC2.detach();
      break;
    case '5':
      ESC2.attach(10);
      sweepServo(ESC2, right_pos, 9);
      ESC2.detach();
      break;
    case 'l':
      unsigned int distance = measureDistance(); //distance in cm
      Serial.println(distance);
      break;
  }
}
