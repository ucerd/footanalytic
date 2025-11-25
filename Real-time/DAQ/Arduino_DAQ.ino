  #include <SoftwareSerial.h>
int a=0;
int D_0=6;
int D_1=7;
int D_2=8;
int D_3=9;
int D_4=10;
int D_5=11;
int D_6=12;
int D_7=13;
int D_8=5;
int D_9=4;
int D_10=3;
int D_11=2;
int D_12=1;
int D_13=0;
int D_14=14;
int D_15=15;

int LED_PIN1= 51;
boolean led_blink=0;
int Marker=8;

int R0_C0=0;
int R0_C1=0;
int R0_C2=0;
int R0_C3=0;
int R0_C4=0;
int R0_C5=0;
int R0_C6=0;
int R0_C7=0;

int R1_C0=0;
int R1_C1=0;
int R1_C2=0;
int R1_C3=0;
int R1_C4=0;
int R1_C5=0;
int R1_C6=0;
int R1_C7=0;

int R2_C0=0;
int R2_C1=0;
int R2_C2=0;
int R2_C3=0;
int R2_C4=0;
int R2_C5=0;
int R2_C6=0;
int R2_C7=0;

int R3_C0=0;
int R3_C1=0;
int R3_C2=0;
int R3_C3=0;
int R3_C4=0;
int R3_C5=0;
int R3_C6=0;
int R3_C7=0;

int R4_C0=0;
int R4_C1=0;
int R4_C2=0;
int R4_C3=0;
int R4_C4=0;
int R4_C5=0;
int R4_C6=0;
int R4_C7=0;

int R5_C0=0;
int R5_C1=0;
int R5_C2=0;
int R5_C3=0;
int R5_C4=0;
int R5_C5=0;
int R5_C6=0;
int R5_C7=0;

int R6_C0=0;
int R6_C1=0;
int R6_C2=0;
int R6_C3=0;
int R6_C4=0;
int R6_C5=0;
int R6_C6=0;
int R6_C7=0;

int R7_C0=0;
int R7_C1=0;
int R7_C2=0;
int R7_C3=0;
int R7_C4=0;
int R7_C5=0;
int R7_C6=0;
int R7_C7=0;

int R8_C0=0;
int R8_C1=0;
int R8_C2=0;
int R8_C3=0;
int R8_C4=0;
int R8_C5=0;
int R8_C6=0;
int R8_C7=0;

int R9_C0=0;
int R9_C1=0;
int R9_C2=0;
int R9_C3=0;
int R9_C4=0;
int R9_C5=0;
int R9_C6=0;
int R9_C7=0;

int R10_C0=0;
int R10_C1=0;
int R10_C2=0;
int R10_C3=0;
int R10_C4=0;
int R10_C5=0;
int R10_C6=0;
int R10_C7=0;

int R11_C0=0;
int R11_C1=0;
int R11_C2=0;
int R11_C3=0;
int R11_C4=0;
int R11_C5=0;
int R11_C6=0;
int R11_C7=0;

int R12_C0=0;
int R12_C1=0;
int R12_C2=0;
int R12_C3=0;
int R12_C4=0;
int R12_C5=0;
int R12_C6=0;
int R12_C7=0;

int R13_C0=0;
int R13_C1=0;
int R13_C2=0;
int R13_C3=0;
int R13_C4=0;
int R13_C5=0;
int R13_C6=0;
int R13_C7=0;

int R14_C0=0;
int R14_C1=0;
int R14_C2=0;
int R14_C3=0;
int R14_C4=0;
int R14_C5=0;
int R14_C6=0;
int R14_C7=0;

int R15_C0=0;
int R15_C1=0;
int R15_C2=0;
int R15_C3=0;
int R15_C4=0;
int R15_C5=0;
int R15_C6=0;
int R15_C7=0;


void setup() {
Serial.begin(115200);
pinMode(D_0, OUTPUT);
pinMode(D_1, OUTPUT);
pinMode(D_2, OUTPUT);
pinMode(D_3, OUTPUT);
pinMode(D_4, OUTPUT);
pinMode(D_5, OUTPUT);
pinMode(D_6, OUTPUT);
pinMode(D_7, OUTPUT);

pinMode(D_8, OUTPUT);
pinMode(D_9, OUTPUT);
pinMode(D_10, OUTPUT);
pinMode(D_11, OUTPUT);
pinMode(D_12, OUTPUT);
pinMode(D_13, OUTPUT);
pinMode(D_14, OUTPUT);
pinMode(D_15, OUTPUT);

pinMode(LED_PIN1, OUTPUT);


}

void loop() {

// Row 1 Read 8 Values 
  digitalWrite(D_0, HIGH);
  delay(10);
  R0_C0 = analogRead(A0);
  R0_C1 = analogRead(A1);
  R0_C2 = analogRead(A2);
  R0_C3 = analogRead(A3);
  R0_C4 = analogRead(A4);
  R0_C5 = analogRead(A5);
  R0_C6 = analogRead(A6);
  R0_C7 = analogRead(A7);
  digitalWrite(D_0, LOW);

// Row 2 Read 8 Values 
  digitalWrite(D_1, HIGH);
  delay(10);
  R1_C0 = analogRead(A0);
  R1_C1 = analogRead(A1);
  R1_C2 = analogRead(A2);
  R1_C3 = analogRead(A3);
  R1_C4 = analogRead(A4);
  R1_C5 = analogRead(A5);
  R1_C6 = analogRead(A6);
  R1_C7 = analogRead(A7);
  digitalWrite(D_1, LOW);
  
// Row 3 Read 8 Values 
  digitalWrite(D_2, HIGH);
  delay(10);
  R2_C0 = analogRead(A0);
  R2_C1 = analogRead(A1);
  R2_C2 = analogRead(A2);
  R2_C3 = analogRead(A3);
  R2_C4 = analogRead(A4);
  R2_C5 = analogRead(A5);
  R2_C6 = analogRead(A6);
  R2_C7 = analogRead(A7);
  digitalWrite(D_2, LOW);

// Row 4 Read 8 Values   
  digitalWrite(D_3, HIGH);
  delay(10);
  R3_C0 = analogRead(A0);
  R3_C1 = analogRead(A1);
  R3_C2 = analogRead(A2);
  R3_C3 = analogRead(A3);
  R3_C4 = analogRead(A4);
  R3_C5 = analogRead(A5);
  R3_C6 = analogRead(A6);
  R3_C7 = analogRead(A7);
  digitalWrite(D_3, LOW);
  

// Row 5 Read 8 Values 
  digitalWrite(D_4, HIGH);
  delay(10);
  R4_C0 = analogRead(A0);
  R4_C1 = analogRead(A1);
  R4_C2 = analogRead(A2);
  R4_C3 = analogRead(A3);
  R4_C4 = analogRead(A4);
  R4_C5 = analogRead(A5);
  R4_C6 = analogRead(A6);
  R4_C7 = analogRead(A7);
  digitalWrite(D_4, LOW);
  
// Row 6 Read 8 Values 
  digitalWrite(D_5, HIGH);
  delay(10);
  R5_C0 = analogRead(A0);
  R5_C1 = analogRead(A1);
  R5_C2 = analogRead(A2);
  R5_C3 = analogRead(A3);
  R5_C4 = analogRead(A4);
  R5_C5 = analogRead(A5);
  R5_C6 = analogRead(A6);
  R5_C7 = analogRead(A7);
  digitalWrite(D_5, LOW);
  
// Row 7 Read 8 Values 
  digitalWrite(D_6, HIGH);
  delay(10);
  R6_C0 = analogRead(A0);
  R6_C1 = analogRead(A1);
  R6_C2 = analogRead(A2);
  R6_C3 = analogRead(A3);
  R6_C4 = analogRead(A4);
  R6_C5 = analogRead(A5);
  R6_C6 = analogRead(A6);
  R6_C7 = analogRead(A7);
  digitalWrite(D_6, LOW);
  
// Row 8 Read 8 Values 
  digitalWrite(D_7, HIGH);
  delay(10);
  R7_C0 = analogRead(A0);
  R7_C1 = analogRead(A1);
  R7_C2 = analogRead(A2);
  R7_C3 = analogRead(A3);
  R7_C4 = analogRead(A4);
  R7_C5 = analogRead(A5);
  R7_C6 = analogRead(A6);
  R7_C7 = analogRead(A7);
  digitalWrite(D_7, LOW);




// Row 9 Read 8 Values 
  digitalWrite(D_8, HIGH);
  delay(10);
  R15_C0=analogRead(A8);
  R15_C1=analogRead(A9);
  R15_C2=analogRead(A10);
  R15_C3=analogRead(A11);
  R15_C4=analogRead(A12);
  R15_C5=analogRead(A13);
  R15_C6=analogRead(A14);
  R15_C7=analogRead(A15);
  digitalWrite(D_8, LOW);

// Row 11 Read 8 Values 
  digitalWrite(D_10, HIGH);
  delay(10);
  R13_C0=analogRead(A8);
  R13_C1=analogRead(A9);
  R13_C2=analogRead(A10);
  R13_C3=analogRead(A11);
  R13_C4=analogRead(A12);
  R13_C5=analogRead(A13);
  R13_C6=analogRead(A14);
  R13_C7=analogRead(A15);
  digitalWrite(D_10, LOW);

// Row 10 Read 8 Values 
  digitalWrite(D_9, HIGH);
  delay(10);
  R14_C0=analogRead(A8);
  R14_C1=analogRead(A9);
  R14_C2=analogRead(A10);
  R14_C3=analogRead(A11);
  R14_C4=analogRead(A12);
  R14_C5=analogRead(A13);
  R14_C6=analogRead(A14);
  R14_C7=analogRead(A15);
  digitalWrite(D_9, LOW);
  
  // Row 12 Read 8 Values 
  digitalWrite(D_11, HIGH);
  delay(10);
  R12_C0=analogRead(A8);
  R12_C1=analogRead(A9);
  R12_C2=analogRead(A10);
  R12_C3=analogRead(A11);
  R12_C4=analogRead(A12);
  R12_C5=analogRead(A13);
  R12_C6=analogRead(A14);
  R12_C7=analogRead(A15);
  digitalWrite(D_11, LOW);

  digitalWrite(D_12, HIGH);
  delay(10);
  R11_C0=analogRead(A8);
  R11_C1=analogRead(A9);
  R11_C2=analogRead(A10);
  R11_C3=analogRead(A11);
  R11_C4=analogRead(A12);
  R11_C5=analogRead(A13);
  R11_C6=analogRead(A14);
  R11_C7=analogRead(A15);
  digitalWrite(D_12, LOW);



// Row 14 Read 8 Values 
  digitalWrite(D_13, HIGH);
  delay(10);
  R10_C0=analogRead(A8);
  R10_C1=analogRead(A9);
  R10_C2=analogRead(A10);
  R10_C3=analogRead(A11);
  R10_C4=analogRead(A12);
  R10_C5=analogRead(A13);
  R10_C6=analogRead(A14);
  R10_C7=analogRead(A15);
  digitalWrite(D_13, LOW);


// Row 16 Read 8 Values 
  digitalWrite(D_14, HIGH);
  delay(10);
  R8_C0=analogRead(A8);
  R8_C1=analogRead(A9);
  R8_C2=analogRead(A10);
  R8_C3=analogRead(A11);
  R8_C4=analogRead(A12);
  R8_C5=analogRead(A13);
  R8_C6=analogRead(A14);
  R8_C7=analogRead(A15);
  digitalWrite(D_14, LOW);


// Row 15 Read 8 Values 
  digitalWrite(D_15, HIGH);
  delay(10);
  R9_C0=analogRead(A8);
  R9_C1=analogRead(A9);
  R9_C2=analogRead(A10);
  R9_C3=analogRead(A11);
  R9_C4=analogRead(A12);
  R9_C5=analogRead(A13);
  R9_C6=analogRead(A14);
  R9_C7=analogRead(A15);
  digitalWrite(D_15, LOW);
  
  
 digitalWrite(LED_PIN1, led_blink);
 led_blink=!led_blink;
//Serial.println("Row 1");
  Serial.println("$$$$");
  Serial.print(R0_C0);Serial.print(",");Serial.print(R0_C1);Serial.print(",");Serial.print(R0_C2);Serial.print(",");Serial.print(R0_C3);Serial.print(",");Serial.print(R0_C4);Serial.print(",");Serial.print(R0_C5);Serial.print(",");Serial.print(R0_C6);Serial.print(",");Serial.println(R0_C7);//Serial.println("\n");
//Serial.println("Row 2");
  Serial.print(R1_C0);Serial.print(",");Serial.print(R1_C1);Serial.print(",");Serial.print(R1_C2);Serial.print(",");Serial.print(R1_C3);Serial.print(",");Serial.print(R1_C4);Serial.print(",");Serial.print(R1_C5);Serial.print(",");Serial.print(R1_C6);Serial.print(",");Serial.println(R1_C7);//Serial.println("\n");
//Serial.println("Row 3");
  Serial.print(R2_C0);Serial.print(",");Serial.print(R2_C1);Serial.print(",");Serial.print(R2_C2);Serial.print(",");Serial.print(R2_C3);Serial.print(",");Serial.print(R2_C4);Serial.print(",");Serial.print(R2_C5);Serial.print(",");Serial.print(R2_C6);Serial.print(",");Serial.println(R2_C7);//Serial.println("\n");
//Serial.println("Row 4");
  Serial.print(R3_C0);Serial.print(",");Serial.print(R3_C1);Serial.print(",");Serial.print(R3_C2);Serial.print(",");Serial.print(R3_C3);Serial.print(",");Serial.print(R3_C4);Serial.print(",");Serial.print(R3_C5);Serial.print(",");Serial.print(R3_C6);Serial.print(",");Serial.println(R3_C7);//Serial.println("\n");
//Serial.println("Row 5");
  Serial.print(R4_C0);Serial.print(",");Serial.print(R4_C1);Serial.print(",");Serial.print(R4_C2);Serial.print(",");Serial.print(R4_C3);Serial.print(",");Serial.print(R4_C4);Serial.print(",");Serial.print(R4_C5);Serial.print(",");Serial.print(R4_C6);Serial.print(",");Serial.println(R4_C7);//Serial.println("\n");
//Serial.println("Row 6");
  Serial.print(R5_C0);Serial.print(",");Serial.print(R5_C1);Serial.print(",");Serial.print(R5_C2);Serial.print(",");Serial.print(R5_C3);Serial.print(",");Serial.print(R5_C4);Serial.print(",");Serial.print(R5_C5);Serial.print(",");Serial.print(R5_C6);Serial.print(",");Serial.println(R5_C7);//Serial.println("\n");
//Serial.println("Row 7");
  Serial.print(R6_C0);Serial.print(",");Serial.print(R6_C1);Serial.print(",");Serial.print(R6_C2);Serial.print(",");Serial.print(R6_C3);Serial.print(",");Serial.print(R6_C4);Serial.print(",");Serial.print(R6_C5);Serial.print(",");Serial.print(R6_C6);Serial.print(",");Serial.println(R6_C7);//Serial.println("\n");
//Serial.println("Row 8");
  Serial.print(R7_C0);Serial.print(",");Serial.print(R7_C1);Serial.print(",");Serial.print(R7_C2);Serial.print(",");Serial.print(R7_C3);Serial.print(",");Serial.print(R7_C4);Serial.print(",");Serial.print(R7_C5);Serial.print(",");Serial.print(R7_C6);Serial.print(",");Serial.println(R7_C7);//Serial.println("\n");
// Secon part of foot
//Serial.println("Row 9");
  Serial.print(R15_C0);Serial.print(",");Serial.print(R15_C1);Serial.print(",");Serial.print(R15_C2);Serial.print(",");Serial.print(R15_C3);Serial.print(",");Serial.print(R15_C4);Serial.print(",");Serial.print(R15_C5);Serial.print(",");Serial.print(R15_C6);Serial.print(",");Serial.println(R15_C7);//Serial.println("\n");
//Serial.println("Row 11");
  Serial.print(R13_C0);Serial.print(",");Serial.print(R13_C1);Serial.print(",");Serial.print(R13_C2);Serial.print(",");Serial.print(R13_C3);Serial.print(",");Serial.print(R13_C4);Serial.print(",");Serial.print(R13_C5);Serial.print(",");Serial.print(R13_C6);Serial.print(",");Serial.println(R13_C7);//Serial.println("\n");
//Serial.println("Row 10");
  Serial.print(R14_C0);Serial.print(",");Serial.print(R14_C1);Serial.print(",");Serial.print(R14_C2);Serial.print(",");Serial.print(R14_C3);Serial.print(",");Serial.print(R14_C4);Serial.print(",");Serial.print(R14_C5);Serial.print(",");Serial.print(R14_C6);Serial.print(",");Serial.println(R14_C7);//Serial.println("\n");
//Serial.println("Row 12");
  Serial.print(R12_C0);Serial.print(",");Serial.print(R12_C1);Serial.print(",");Serial.print(R12_C2);Serial.print(",");Serial.print(R12_C3);Serial.print(",");Serial.print(R12_C4);Serial.print(",");Serial.print(R12_C5);Serial.print(",");Serial.print(R12_C6);Serial.print(",");Serial.println(R12_C7);//Serial.println("\n");
// Working fine till here
//Serial.println("Row 13");
// iisue with this line
  Serial.print(R11_C0);Serial.print(",");Serial.print(R11_C1);Serial.print(",");Serial.print(R11_C2);Serial.print(",");Serial.print(R11_C3);Serial.print(",");Serial.print(R11_C4);Serial.print(",");Serial.print(R11_C5);Serial.print(",");Serial.print(R11_C6);Serial.print(",");Serial.println(R11_C7);//Serial.println("\n");
//Serial.println("Row 14");
  Serial.print(R10_C0);Serial.print(",");Serial.print(R10_C1);Serial.print(",");Serial.print(R10_C2);Serial.print(",");Serial.print(R10_C3);Serial.print(",");Serial.print(R10_C4);Serial.print(",");Serial.print(R10_C5);Serial.print(",");Serial.print(R10_C6);Serial.print(",");Serial.println(R10_C7);//Serial.println("\n");
//working good from here
//Serial.println("Row 16");
  Serial.print(R8_C0);Serial.print(",");Serial.print(R8_C1);Serial.print(",");Serial.print(R8_C2);Serial.print(",");Serial.print(R8_C3);Serial.print(",");Serial.print(R8_C4);Serial.print(",");Serial.print(R8_C5);Serial.print(",");Serial.print(R8_C6);Serial.print(",");Serial.println(R8_C7);//Serial.println("\n"); 
//Serial.println("Row 15");
  Serial.print(R9_C0);Serial.print(",");Serial.print(R9_C1);Serial.print(",");Serial.print(R9_C2);Serial.print(",");Serial.print(R9_C3);Serial.print(",");Serial.print(R9_C4);Serial.print(",");Serial.print(R9_C5);Serial.print(",");Serial.print(R9_C6);Serial.print(",");Serial.println(R9_C7);//Serial.println("\n");
  delay(100);

}
