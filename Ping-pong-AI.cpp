#include <SPI.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
Adafruit_SSD1306 display = Adafruit_SSD1306(128, 32, &Wire);

//---------------TEST MODE----------------------
#define TEST 0
//---------------TEST MODE----------------------



#define leftButton 26
#define rightButton 27


int startX = 64;
int startY = 16;
bool vecX = 0;
bool vecY = 0;
int ballX = 0;
int ballY = 0;
int score = 0;
bool gameOver = 0;
int ballPrevX = 0;
int ballPrevY = 0;
int ballSpeedX = 0;
int ballSpeedY = 0;
int predictedY = 0;
#define ballSpeed 30


#define wallSize 6
#define wallStartY 16
int wallY = wallStartY-wallSize/2;

#define botWallSize 6
#define botWallStartY 16
#define botWallSpeed 35
int botWallY = botWallStartY-botWallSize/2;
uint64_t tmr = 0;
uint64_t tmr1 = 0;
uint64_t tmr2 = 0;
int fps = 0;
int fpsShow = 0;








//---------------AI------------------------------------------
#define inputLayer 3
#define hidenLayer 4
#define outputLayer 2
#define discountFactor 0.9
#define epsilon 0.1

float input[inputLayer];
float hiden[hidenLayer];
float output[outputLayer];

float inputWeight[inputLayer][hidenLayer];    //веса вход-> скрытый слой
float hidenWeight[hidenLayer][outputLayer];    //веса скрытый-> выход слой

//--------BackProp----------------------------------

//float deltaPart[][];
//float deltaLast[][];
//float deltaOther[][];
//float delta[][];

float reward = 0;
bool action = 0;
bool oldAction = 0;

float relu(float x) { return (x > 0) ? x : 0; }

#define learningRate 0.1


//--------BackProp----------------------------------

float randomFloat() {
  uint64_t rnd;
  for (int i = 0; i < 16; i++) {
    rnd *= 6;
    rnd += analogRead(A0) & 3;
  }
  float rndFloat;
  rnd %= 200;
  rndFloat = (float)rnd;
  return (rndFloat/100.0)-1.0;
}

void setupAI() {
  if (TEST) {
    Serial.println("inputWeight");
  }
  for (int i = 0; i < inputLayer; i++) {
    for (int k = 0; k < hidenLayer; k++) {
      inputWeight[i][k] = randomFloat();
      if (TEST) {
        Serial.print(inputWeight[i][k]);
        Serial.print(" ");
        Serial.print(i);
        Serial.print(":");
        Serial.print(k);
        Serial.print(" ");
      }
    }
    if (TEST) {
      Serial.println();
    }
  }
  if (TEST) {
    Serial.println("hidenWeight");
  }
  for (int n = 0; n < hidenLayer; n++) {
    for (int j = 0; j < outputLayer; j++) {
      hidenWeight[n][j] = randomFloat();
      if (TEST) {
        Serial.print(hidenWeight[n][j]);
        Serial.print(" ");
        Serial.print(n);
        Serial.print(":");
        Serial.print(j);
        Serial.print(" ");
      }
    }
    if (TEST) {
      Serial.println();
    }
  }
  return;
}

void AI() {
  if (TEST) {
    Serial.println("AI ");
  }
  input[0] = (float)(ballX/128.0);
  if (TEST) {
    Serial.print(input[0]);
    Serial.print(" ");
  }
  input[1] = (float)(ballY/32.0);
  if (TEST) {
    Serial.print(input[1]);
    Serial.print(" ");
  }
  input[2] = (float)(botWallY/32.0);
  if (TEST) {
    Serial.print(input[2]);
    Serial.println();
  }

  for (int i = 0; i < hidenLayer; i++) {  // i - нейроны скрытого слоя
    hiden[i] = 0.0;
    for (int j = 0; j < inputLayer; j++) {  // j - нейроны входного слоя
      hiden[i] += input[j] * inputWeight[j][i];  // правильная индексация!
    }
    hiden[i] = relu(hiden[i]);
  } 

  for (int i = 0; i < outputLayer; i++) {
    output[i] = 0;
    for (int j = 0; j < hidenLayer; j++) {
      output[i] += hiden[j] * hidenWeight[j][i];
    }
    // Добавить softmax или сигмоиду для вероятностей!
    output[i] = 1.0 / (1.0 + exp(-output[i]));  // сигмоида
  }
  if (TEST) {
    Serial.println();
    Serial.print(output[0]);
    Serial.print(" ");
    Serial.print(output[1]);
  }
  if (output[0] > output[1]) {
    action = true;
  } else {
    action = false;
  }
  return;
}



void trainAI() {
  // 1. Вычисляем дельты для выходного слоя
  float deltaOutput[outputLayer];
  
  for (int i = 0; i < outputLayer; i++) {
    if (i == action) {
      deltaOutput[i] = -reward * (1.0 - output[i]);
    } else {
      deltaOutput[i] = 0.0;
    }
  }

  // Остальной код без изменений...
  // 2. Обновляем веса hidenWeight
  Serial.println("hidenWeight");
  for (int i = 0; i < outputLayer; i++) {
    for (int j = 0; j < hidenLayer; j++) {
      float gradient = deltaOutput[i] * hiden[j];
      hidenWeight[j][i] = hidenWeight[j][i] - learningRate * gradient;
      Serial.print(hidenWeight[j][i]);
      Serial.print(" ");
    }
    Serial.println();
  }

  // 3. Обновляем веса inputWeight
  Serial.println("inputWeight");
  for (int i = 0; i < hidenLayer; i++) {
    float sum = 0.0;
    for (int k = 0; k < outputLayer; k++) {
      sum += deltaOutput[k] * hidenWeight[i][k];
    }
    
    float deltaHiden = sum * hiden[i] * (1 - hiden[i]);
    
    for (int j = 0; j < inputLayer; j++) {
      float gradient = deltaHiden * input[j];
      inputWeight[j][i] = inputWeight[j][i] - learningRate * gradient;
      Serial.print(inputWeight[j][i]);
      Serial.print(" ");
    }
    Serial.println();
  }
  Serial.println();
  return;
}


//---------------AI------------------------------------------


void setup() {
  pinMode(leftButton, INPUT);
  pinMode(rightButton, INPUT);
  display.begin(SSD1306_SWITCHCAPVCC, 0x3C);
  display.setTextColor(SSD1306_WHITE);
  Serial.begin(9600);
  if (TEST) Serial.println(randomFloat());
  setupAI();
  display.clearDisplay();
  display.display();
  vecX = randomBool();
  vecY = randomBool();
  ballX = startX;
  ballY = startY;
  ballPrevX = ballX;
  ballPrevY = ballY;
}

void loop() {
  while (!gameOver) {
    AI();
    display.clearDisplay();
    display.setCursor(59, 11);
    if (digitalRead(leftButton)==1){ 
      if (wallY > 0) {wallY--; }
    }else if (digitalRead(rightButton)==1){
      if (wallY+wallSize < 32) { wallY++;  }
    }

    if (millis()-tmr2 >= ballSpeed) {
      tmr2 = millis();
      if (vecX == 1) {ballX = ballX + 1;
      }else{ballX = ballX - 1;}
      if (vecY == 1) { ballY = ballY + 1; 
      }else{ ballY = ballY - 1; }

      if (ballY <= 0) {  // Было <= 1
        vecY = !vecY;
        ballY = 0; // Фиксируем позицию
      }else if (ballY >= 31) {  // Было >= 30 (31 - последняя строка)
        vecY = !vecY;
        ballY = 31; // Фиксируем позицию
      }
      if (ballX <= 2) {
        if (ballY >= wallY && ballY <= wallY + wallSize) { // Проверяем всю высоту ракетки
          vecX = !vecX;
          ballX = 2; // Фиксируем позицию
        } else {
          reward = 1.0;
          trainAI();
          gameOver = 1;
        }
      }
      if (ballX >= 125) {
        if (ballY >= botWallY && ballY <= botWallY + botWallSize) {
          reward = 1.0;
          trainAI();
          vecX = !vecX;
          ballX = 125;
        } else {
          score++;
          reward = -1.0;
          trainAI();
          vecX = randomBool();
          vecY = randomBool();
          ballX = startX;
          ballY = startY;
        }
      }
    }

    ballSpeedX = ballX - ballPrevX;
    ballSpeedY = ballY - ballPrevY;
    ballPrevX = ballX;
    ballPrevY = ballY;

    display.fillCircle(ballX, ballY, 1, SSD1306_WHITE);                                     //отрисовываем шар
    display.drawLine(0, wallY, 0, wallY+wallSize, SSD1306_WHITE);              //отрисовываем нашу стенку
    display.drawLine(127, botWallY, 127, botWallY+botWallSize, SSD1306_WHITE);              //отрисовываем стенку противника
    display.drawLine(0, 32, 127, 32, SSD1306_WHITE);
    display.print(score);
    if (TEST == 1) {
      display.setCursor(59, 19);
      display.print(fpsShow);
      if (millis()-tmr >= 1000) {
        tmr = millis();
        Serial.println(fps);
        fpsShow = fps;
        fps = 0;
      }
    }
    display.display();
    fps++;

    if (millis()-tmr1 >= botWallSpeed) {
      tmr1 = millis();
      if (action) {
        if (botWallY+wallSize < 32) { botWallY++;  }
      }else{
        if (botWallY > 0) {botWallY--; }
      }
    }
  }
  vecX = randomBool();
  vecY = randomBool();
  ballX = startX;
  ballY = startY;
  display.clearDisplay();
  display.setCursor(40, 3);
  display.print("GAME OVER!");
  display.setCursor(30, 13);
  display.print("Your score: ");
  display.print(score);
  score = 0;
  display.display(); 
  delay(1500);
  gameOver = 0;
}

bool randomBool() {
  uint32_t seed = 0;
  for (int i = 0; i < 16; i++) {
    seed *= 4;
    seed += analogRead(A0) & 3;
    randomSeed(seed);
  }
  if (seed >= 2000000000) {return 1; }else{ return 0; }
}
