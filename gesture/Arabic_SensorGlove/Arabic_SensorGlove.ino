#include <Wire.h>
#include <MPU6050.h>
#include <LiquidCrystal_I2C.h>

LiquidCrystal_I2C lcd(0x27, 16, 2);

// ─── Pin Definitions ────────────────────────────────────────
const int FLEX_PINS[5] = {34, 35, 33, 32, 36};

// ─── Calibration Storage ────────────────────────────────────
int openVal[5]   = {0};
int closedVal[5] = {0};
int threshold[5] = {0};

// ─── MPU6050 ────────────────────────────────────────────────
MPU6050 mpu;
int16_t ax, ay, az, gx, gy, gz;

// ─── Gesture Struct ─────────────────────────────────────────
struct Gesture {
  const char* arabic;
  const char* english;
  int fingers[5];
  int pitchMin;
  int pitchMax;
};

// ─── Gesture Table ───────────────────────────────────────────
const Gesture GESTURES[] = {
  { "أ",    "Alef",   {1, 0, 0, 0, 0},  100,    200   },
  { "ب",    "Ba",     {0, 1, 0, 0, 0},  30,   200   },
  { "ت",    "Ta",     {0, 1, 1, 0, 0},  50,    90   },
  { "ث",    "Tha",    {0, 1, 1, 1, 0},  -999, -999 },
 // { "ج",    "Jeem",   {1, 1, 1, 1, 1},  20,   80   },
//  { "د",    "Dal",    {0, 1, 0, 0, 0},  20,   60   },
 // { "ذ",    "Dhal",   {1, 1, 1, 0, 0},  61,   70   },
//  { "ر",    "Ra",     {0, 1, 0, 0, 0},  50,   999  },
//  { "ز",    "Zay",    {0, 1, 0, 0, 1},  -999, -999 },
 // { "ط",    "Tah",    {0, 1, 0, 0, 0},  30,   80   },
 // { "ظ",    "Dha",    {1, 1, 1, 0, 0},  70,   90   },
  { "ع",    "Ain",    {0, 1, 1, 0, 0},  -90,   20  },
  { "غ",    "Ghain",  {1, 1, 1, 0, 0},  -90,   200   },
  { "ك",    "Kaf",    {0, 1, 1, 1, 1},  -999, 999  },
  { "ل",    "Lam",    {1, 1, 0, 0, 0},  20, 80 },
  { "م",    "Meem",   {0, 0, 0, 0, 1},  30,   100  },
  { "ن",    "Noon",   {1, 1, 0, 0, 0},  100,  190 },
  { "و",    "Waw",    {1, 0, 0, 0, 0},  -90,  10  },
  { "ي",    "Ya",     {1, 0, 0, 0, 1},  -999, -999 },
  { "مرحبا","Hello",  {1, 1, 1, 1, 1},  50,    89 },
  { "نعم",  "Yes",    {0, 0, 0, 0, 0},  10,   80   },
  { "لا",   "No",     {1, 0, 1, 0, 0},  30,   60   },
  { "شكرا", "Thanks", {1, 1, 1, 1, 1},  90,  150  },
  { "جيد",  "Good",   {1, 0, 0, 0, 0},  0   },
  { "تفضل", "Please", {1, 1, 1, 1, 1},  10,   40   },
};

const int NUM_GESTURES = sizeof(GESTURES) / sizeof(GESTURES[0]);

// ================= ARABIC LETTERS =================

void Aleef() {
  byte c[8] = {B00100,B00100,B00100,B00100,B00100,B00100,B00100,B00100};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Ba() {
  byte c[8] = {B10001,B10001,B10001,B10001,B11111,B00100,B00000,B00000};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Ta() {
  byte c[8] = {B10001,B10001,B11011,B10001,B11111,B00000,B00000,B00000};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Tha() {
  byte c[8] = {B10001,B10101,B11111,B10001,B11111,B00000,B00000,B00000};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Jeem() {
  byte c[8] = {B00000,B00100,B01111,B10010,B00100,B01000,B10000,B11111};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Dal() {
  byte c[8] = {B00000,B01000,B00100,B00010,B00001,B00001,B10001,B11111};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Tah() {
  byte c[8] = {B00100,B00100,B00100,B00100,B00100,B00111,B01001,B11111};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Dha() {
  byte c[8] = {B00100,B00100,B00100,B00110,B00100,B00111,B01001,B11111};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Ain() {
  byte c[8] = {B00000,B01111,B10000,B11100,B01000,B10000,B10000,B11111};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Ghain() {
  byte c[8] = {B00100,B01111,B10000,B11100,B01000,B10000,B10000,B11111};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Kaf() {
  byte c[8] = {B00001,B00001,B00111,B00101,B01101,B10001,B10001,B11111};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Lam() {
  byte c[8] = {B00001,B00001,B00001,B00001,B00001,B10001,B10001,B11111};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Meem() {
  byte c[8] = {B00000,B01111,B01001,B11111,B10000,B10000,B10000,B10000};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Noon() {
  byte c[8] = {B00000,B00000,B00000,B10001,B10101,B10001,B10001,B11111};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Waw() {
  byte c[8] = {B00000,B00001,B00011,B00101,B01111,B00001,B00001,B11111};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void Ya() {
  byte c[8] = {B00011,B00100,B01000,B11111,B00001,B00001,B11111,B01010};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

void La() {
  byte c[8] = {B00001,B10001,B01001,B00101,B00011,B00111,B01001,B11111};
  lcd.createChar(0, c);
  lcd.clear();
  lcd.setCursor(7, 0); lcd.write(0);
}

// ================= WORDS =================

void showHello() {
  byte one[8]   = {B10000,B10000,B10000,B10000,B10000,B11111,B00000,B00000};
  byte two[8]   = {B00010,B00010,B00010,B00010,B00010,B11111,B01000,B00000};
  byte three[8] = {B00000,B00000,B00110,B00001,B00010,B11100,B00000,B00000};
  byte four[8]  = {B00000,B01000,B00100,B00010,B00010,B00011,B00100,B11000};
  byte five[8]  = {B00000,B00000,B00000,B00011,B00101,B11111,B00000,B00000};
  lcd.createChar(0, one);
  lcd.createChar(1, two);
  lcd.createChar(2, three);
  lcd.createChar(3, four);
  lcd.createChar(4, five);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write(0); lcd.write(1); lcd.write(2); lcd.write(3); lcd.write(4);
}

void showPlease() {
  byte one[8]   = {B00010,B00010,B00010,B00010,B00010,B00011,B10010,B11110};
  byte two[8]   = {B00010,B00111,B00101,B00111,B01001,B11111,B00000,B00000};
  byte three[8] = {B00010,B00000,B00111,B00101,B00101,B11111,B00000,B00000};
  byte four[8]  = {B00000,B00001,B01101,B00001,B00001,B11111,B00000,B00000};
  lcd.createChar(0, one);
  lcd.createChar(1, two);
  lcd.createChar(2, three);
  lcd.createChar(3, four);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write(0); lcd.write(1); lcd.write(2); lcd.write(3);
}

void showYes() {
  byte one[8]   = {B00000,B01110,B01010,B01010,B11111,B10000,B10000,B10000};
  byte two[8]   = {B00000,B00000,B00100,B01010,B11111,B00000,B00000,B00000};
  byte three[8] = {B00000,B00100,B00001,B00001,B11111,B00000,B00000,B00000};
  lcd.createChar(0, one);
  lcd.createChar(1, two);
  lcd.createChar(2, three);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write(0); lcd.write(1); lcd.write(2);
}

void showGood() {
  byte one[8]   = {B00000,B00000,B00100,B00010,B00001,B10001,B10001,B01110};
  byte two[8]   = {B00000,B00000,B00000,B00001,B00001,B11111,B00000,B01010};
  byte three[8] = {B00100,B00000,B01100,B00010,B00001,B11110,B00000,B00000};
  lcd.createChar(0, one);
  lcd.createChar(1, two);
  lcd.createChar(2, three);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write(0); lcd.write(1); lcd.write(2);
}

void showThanks() {
  byte one[8]   = {B10000,B10000,B10000,B10000,B10000,B10000,B10000,B10000};
  byte two[8]   = {B00000,B01000,B00100,B00010,B00010,B00011,B00010,B11100};
  byte three[8] = {B00111,B01000,B00100,B00010,B00001,B11111,B00000,B00000};
  byte four[8]  = {B00000,B00000,B00000,B00000,B00000,B11111,B00000,B00000};
  byte five[8]  = {B00100,B00000,B10101,B10101,B10101,B11111,B00000,B00000};
  lcd.createChar(0, one);
  lcd.createChar(1, two);
  lcd.createChar(2, three);
  lcd.createChar(3, four);
  lcd.createChar(4, five);
  lcd.clear();
  lcd.setCursor(0, 0);
  lcd.write(0); lcd.write(1); lcd.write(2); lcd.write(3); lcd.write(4);
}

// ─── LCD Gesture Dispatcher ──────────────────────────────────
void displayGestureOnLCD(const Gesture* g) {
  String eng = String(g->english);
  if      (eng == "Alef")   { Aleef();      }
  else if (eng == "Ba")     { Ba();         }
  else if (eng == "Ta")     { Ta();         }
  else if (eng == "Tha")    { Tha();        }
  else if (eng == "Jeem")   { Jeem();       }
  else if (eng == "Dal")    { Dal();        }
  else if (eng == "Tah")    { Tah();        }
  else if (eng == "Dha")    { Dha();        }
  else if (eng == "Ain")    { Ain();        }
  else if (eng == "Ghain")  { Ghain();      }
  else if (eng == "Kaf")    { Kaf();        }
  else if (eng == "Lam")    { Lam();        }
  else if (eng == "Meem")   { Meem();       }
  else if (eng == "Noon")   { Noon();       }
  else if (eng == "Waw")    { Waw();        }
  else if (eng == "Ya")     { Ya();         }
  else if (eng == "No")     { La();         }
  else if (eng == "Hello")  { showHello();  }
  else if (eng == "Yes")    { showYes();    }
  else if (eng == "Thanks") { showThanks(); }
  else if (eng == "Good")   { showGood();   }
  else if (eng == "Please") { showPlease(); }
  else {
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print(g->arabic);
    lcd.setCursor(0, 1); lcd.print(g->english);
  }
}

// ─── MPU: Get Pitch ──────────────────────────────────────────
float getPitch() {
  mpu.getMotion6(&ax, &ay, &az, &gx, &gy, &gz);
  return atan2((float)ay, (float)az) * 180.0 / PI;
}

// ─── Read one flex sensor (average 5 samples) ────────────────
int readFinger(int idx) {
  long sum = 0;
  for (int i = 0; i < 5; i++) sum += analogRead(FLEX_PINS[idx]);
  return sum / 5;
}

// ─── Get binary state for one finger ────────────────────────
int getFingerState(int idx) {
  return (readFinger(idx) >= threshold[idx]) ? 1 : 0;
}

// ─── Calibration ─────────────────────────────────────────────
void calibrate() {
  // ── Calibration Started ──
  lcd.clear();
  lcd.setCursor(2, 0); lcd.print("Calibration");
  lcd.setCursor(4, 1); lcd.print("Started...");
  Serial.println("===== CALIBRATION STARTING =====");
  delay(2000);

  // ── Open Hand ──
  lcd.clear();
  lcd.setCursor(3, 0); lcd.print("Open Hand");
  lcd.setCursor(2, 1); lcd.print("Hold still...");
  Serial.println("[1/2] Spread hand FULLY OPEN and hold...");
  for (int c = 5; c >= 1; c--) {
    lcd.setCursor(13, 1); lcd.print(c); lcd.print("s");
    Serial.printf("  %d...\n", c);
    delay(1000);
  }

  long sums[5] = {0};
  for (int s = 0; s < 50; s++) {
    for (int i = 0; i < 5; i++) sums[i] += analogRead(FLEX_PINS[i]);
    delay(20);
  }
  for (int i = 0; i < 5; i++) openVal[i] = sums[i] / 50;
  Serial.printf("  Open: Th=%d Ix=%d Md=%d Rg=%d Pk=%d\n",
    openVal[0],openVal[1],openVal[2],openVal[3],openVal[4]);

  // ── Close Hand ──
  lcd.clear();
  lcd.setCursor(2, 0); lcd.print("Close Hand");
  lcd.setCursor(2, 1); lcd.print("Hold still...");
  Serial.println("[2/2] Make a tight FIST and hold...");
  for (int c = 5; c >= 1; c--) {
    lcd.setCursor(13, 1); lcd.print(c); lcd.print("s");
    Serial.printf("  %d...\n", c);
    delay(1000);
  }

  memset(sums, 0, sizeof(sums));
  for (int s = 0; s < 50; s++) {
    for (int i = 0; i < 5; i++) sums[i] += analogRead(FLEX_PINS[i]);
    delay(20);
  }
  for (int i = 0; i < 5; i++) closedVal[i] = sums[i] / 50;
  Serial.printf("  Fist: Th=%d Ix=%d Md=%d Rg=%d Pk=%d\n",
    closedVal[0],closedVal[1],closedVal[2],closedVal[3],closedVal[4]);

  // ── Compute Thresholds ──
  const char* names[] = {"Thumb","Index","Middle","Ring","Pinky"};
  for (int i = 0; i < 5; i++) {
    threshold[i] = (openVal[i] + closedVal[i]) / 2;
    Serial.printf("  %s: open=%d fist=%d threshold=%d\n",
      names[i], openVal[i], closedVal[i], threshold[i]);
  }

  // ── Done ──
  lcd.clear();
  lcd.setCursor(3, 0); lcd.print("Calibration");
  lcd.setCursor(5, 1); lcd.print("Done!");
  Serial.println("===== CALIBRATION DONE! Start signing... =====\n");
  delay(1500);
  lcd.clear();
}

// ─── Gesture Matching ────────────────────────────────────────
const Gesture* matchGesture(int f[5], float pitch) {
  int bestScore = -1;
  int bestIdx   = -1;

  for (int g = 0; g < NUM_GESTURES; g++) {
    if (GESTURES[g].pitchMin != -999) {
      if (pitch < GESTURES[g].pitchMin || pitch > GESTURES[g].pitchMax) continue;
    }
    int score = 0;
    bool valid = true;
    for (int i = 0; i < 5; i++) {
      if (GESTURES[g].fingers[i] == 2) { score++; continue; }
      if (GESTURES[g].fingers[i] == f[i]) score++;
      else { valid = false; break; }
    }
    if (valid && score > bestScore) {
      bestScore = score;
      bestIdx   = g;
    }
  }
  return (bestIdx >= 0) ? &GESTURES[bestIdx] : nullptr;
}

// ─── Setup ───────────────────────────────────────────────────
void setup() {
  Serial.begin(115200);
  delay(500);

  Wire.begin(21, 22);
  mpu.initialize();
  if (mpu.testConnection())
    Serial.println("MPU6050 connected");
  else
    Serial.println("MPU6050 NOT found! Check SDA=21 SCL=22");

  lcd.init();
  lcd.backlight();
  lcd.clear();

  analogReadResolution(12);
  analogSetAttenuation(ADC_11db);

  Serial.println("Arabic Sign Language ESP32 v4");
  calibrate();
}

// ─── Main Loop ───────────────────────────────────────────────
void loop() {
  int raw[5], state[5];
  for (int i = 0; i < 5; i++) {
    raw[i]   = readFinger(i);
    state[i] = getFingerState(i);
  }
  float pitch = getPitch();

  Serial.println("─────────────────────────────────────");
  Serial.printf("RAW  | Th:%4d Ix:%4d Md:%4d Rg:%4d Pk:%4d\n",
    raw[0], raw[1], raw[2], raw[3], raw[4]);
  Serial.printf("STATE| Th:%d Ix:%d Md:%d Rg:%d Pk:%d  (1=open 0=closed)\n",
    state[0], state[1], state[2], state[3], state[4]);
  Serial.printf("PITCH| %.1f deg\n", pitch);

  const Gesture* g = matchGesture(state, pitch);
  if (g) {
    Serial.printf("MATCH| %s (%s)\n", g->arabic, g->english);
    displayGestureOnLCD(g);
    delay(2000);
  } else {
    Serial.println("MATCH| -- No Match --");
    lcd.clear();
    lcd.setCursor(0, 0); lcd.print("No Match...");
    lcd.setCursor(0, 1); lcd.print("Keep signing");
  }

  delay(350);
}