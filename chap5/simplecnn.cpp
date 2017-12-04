/*
 * simplecnn.cpp
 *
 * Purpose: 機械学習と深層学習, chap4, 畳み込みニューラルネットの基本構造
 * Created by: Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>
 */

#include <iostream>
#include <stdio.h>
#include <math.h>

const int INPUTSIZE = 11;    // 入力数
const int FILTERSIZE = 3;    // フィルタの大きさ
const int FILTERNO = 2;      // フィルタの個数
const int POOLSIZE = 3;      // プーリングサイズ
const int POOLOUTSIZE = 3;   // プーリングの出力サイズ
const int MAXINPUTNO = 100;  // 学習データの最大個数
const int SEED = 65535;      // 乱数のシード
const double LIMIT = 0.001;  // 誤差の上限値
const double BIGNUM = 100;   // 誤差の初期値
const int HIDDENNO = 3;      // 中間層のセル数
const double ALPHA = 10;     // 学習係数


void conv(double filter[FILTERSIZE][FILTERSIZE],
          double e[][INPUTSIZE], double convout[][INPUTSIZE]);  // 畳み込みの計算
double calcconv(double filter[][FILTERSIZE],
                double e[][INPUTSIZE], int i, int j);  // フィルタの適用
void pool(double convout[][INPUTSIZE], double poolout[][POOLOUTSIZE]);  // プーリングの計算
double maxpooling(double convout[][INPUTSIZE], int i, int j);  // 最大値プーリング
int getdata(double e[][INPUTSIZE][INPUTSIZE], int r[]);  // データ読み込み
void showdata(double e[][INPUTSIZE][INPUTSIZE], int t[], int n_of_e);  // データ表示
void initfilter(double filter[FILTERNO][FILTERSIZE][FILTERSIZE]);  // フィルタの初期化
double drnd();  // 乱数生成
double f(double u);  // シグモイド関数
void initwh(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1]);  // 中間層の初期化
void initwo(double wo[HIDDENNO + 1]);  // 出力層の初期化
double forward(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1],
               double wo[HIDDENNO + 1], double hi[],
               double e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1]);  // 順方向の計算
void olearn(double wo[HIDDENNO + 1], double hi[],
            double e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double o);  // 出力層の学習
void hlearn(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1],
            double wo[HIDDENNO + 1], double hi[],
            double e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double o);  // 中間層の学習
double f(double u);  // シグモイド関数
void print(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1],
           double wo[HIDDENNO + 1]);  // 結果の出力


int main()
{
  double filter[FILTERNO][FILTERSIZE][FILTERSIZE];  // フィルタの個数
  double e[MAXINPUTNO][INPUTSIZE][INPUTSIZE];  // 入力データ
  int t[MAXINPUTNO];  // 教師データ
  double convout[INPUTSIZE][INPUTSIZE] = { 0 };  // 畳み込み出力
  double poolout[POOLOUTSIZE][POOLOUTSIZE];  // プール出力
  int i, j, m, n;
  int n_of_e;  // 学習データの個数
  double err = BIGNUM;  // 誤差の評価

  int count = 0;
  double ef[POOLSIZE * POOLSIZE * FILTERNO + 1];  // 全結合層への入力データ
  double o;  // 最終出力
  double hi[HIDDENNO + 1];  // 中間層の出力
  double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1];  // 中間層の重み
  double wo[HIDDENNO + 1];

  srand(SEED);

  // フィルターの初期化
  initfilter(filter);

  // 全結合層の重みの初期化
  initwh(wh);  // 中間層の重みの初期化
  initwo(wo);  // 出力層の重みの初期化

  // 入力データの読み込み
  n_of_e = getdata(e, t);
  showdata(e, t, n_of_e);

  // 学習
  while (err > LIMIT) {
    err = 0.0;
    for (i = 0; i < n_of_e; ++i) {  // 学習データごとの繰り返し
      for (j = 0; j < FILTERNO; ++j) {  // フィルタごとの繰り返し
        conv(filter[j], e[i], convout);  //畳み込み計算
        pool(convout, poolout);  // プーリングの計算
        // プーリング出力を全結合相の入力へコピー
        for (m = 0; m < POOLOUTSIZE; ++m) {
          for (n = 0; n < POOLSIZE; ++n) {
            ef[j * POOLOUTSIZE * POOLOUTSIZE + POOLOUTSIZE * m + n] = poolout[m][n];
          }
        }
        ef[POOLOUTSIZE * POOLOUTSIZE * FILTERNO] = t[i];  // 教師データ
      }

      // 順方向の計算
      o = forward(wh, wo, hi, ef);
      // 出力層の重みの調整
      olearn(wo, hi, ef, o);
      // 中間層の重みの調整
      hlearn(wh, wo, hi, ef, o);
      // 誤差の積算
      err += (o - t[i]) * (o - t[i]);
    }
    ++count;
    // 誤差の出力
    std::cout << count << ": " << err << "\n";
  }  // 学習終了

  std::cout << "\n" << "**Results**" << "\n";
  // 結合荷重の出力
  std::cout << "Wheights" << "\n";
  print(wh, wo);
  std::cout << "\n";

  // 教師データに対する出力
  std::cout << "Network output" << "\n";
  std::cout << "teacher -> output" << "\n";
  for (i = 0; i < n_of_e; ++i) {
    std::cout << i << ": " << t[i] << " ";
    for (j = 0; j < FILTERNO; ++j) {  // フィルタごとの繰り返し
      conv(filter[j], e[i], convout);  // 畳み込みの計算
      pool(convout, poolout);  // プーリングの計算
      // プーリング出力を全結合層の入力へコピー
      for (m = 0; m < POOLOUTSIZE; ++m) {
        for (n = 0; n < POOLOUTSIZE; ++n) {
          ef[j * POOLOUTSIZE * POOLOUTSIZE + POOLOUTSIZE * m + n] = poolout[m][n];
        }
      }
      ef[POOLOUTSIZE * POOLOUTSIZE * FILTERNO] = t[i];  // 教師データ
    }
    o = forward(wh, wo, hi, ef);
    std::cout << o << "\n";
  }

  return 0;
}

void print(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1],
           double wo[HIDDENNO + 1])
{
  int i, j;

  for (i = 0; i < HIDDENNO; ++i) {
    for (j = 0; j < POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1; ++j) {
      std::cout << wh[i][j] << " ";
    }
  }
  std::cout << "\n";
  for (i = 0; i < HIDDENNO + 1; ++i) {
    std::cout << wo[i] << " ";
  }
  std::cout << "\n";
}

void hlearn(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1],
            double wo[HIDDENNO + 1],
            double hi[], double e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1],
            double o)
{
  int i, j;
  double dj;  // 中間層の重み計算に利用
  for (j = 0; j < HIDDENNO; ++j) {  // 中間層の各セルjを対象
    dj = hi[j] * (1 - hi[j]) * wo[j]
      * (e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO] - o) * o * (1 - o);
    for (i = 0; i < POOLOUTSIZE * POOLOUTSIZE * FILTERNO; ++i) {
      wh[j][i] += ALPHA * e[i] * dj;  // i番目の重みを処理
    }
    wh[j][i] += ALPHA * (-1.0) * dj;  // 閾値の学習
  }
}


// 出力層の重み学習
void olearn(double wo[HIDDENNO + 1],
            double hi[], double e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1], double o)
{
  int i;
  double d;

  d = (e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO] - o) * o * (1 - o);  // 誤差の計算
  for (i = 0; i < HIDDENNO; ++i) {
    wo[i] += ALPHA * hi[i] * d;  // 重みの学習
  }
  wo[i] += ALPHA * (-1.0) * d;  // 閾値の学習
}


// 順方向の計算
double forward(double wh[HIDDENNO][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1],
               double wo[HIDDENNO + 1], double hi[],
               double e[POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1])
{
  int i, j;
  double u;  // 重み付き和の計算
  double o;  // 出力の計算

  // hiの計算
  for (i = 0; i < HIDDENNO; ++i) {
    u = 0;  // 重み付き和を求める
    for (j = 0; j < POOLOUTSIZE * POOLOUTSIZE * FILTERNO; ++j) {
      u += e[j] * wh[i][j];
    }
    u -= wh[i][j];  // 閾値の計算
    hi[i] = f(u);
  }

  // 出力oの計算
  o = 0;
  for (i = 0; i < HIDDENNO; ++i) {
    o += hi[i] * wo[i];
  }
  o -= wo[i];  // 閾値の処理

  return f(o);
}


// 中間層の重みの初期化
void initwh(double wh[][POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1]) {
  int i, j;

  for (i = 0; i < HIDDENNO; ++i) {
    for (j = 0; j < POOLOUTSIZE * POOLOUTSIZE * FILTERNO + 1; ++j) {
      wh[i][j] = drnd();
    }
  }
}

// 出力層の重みの初期化
void initwo(double wo[]) {
  int i;

  for (i = 0; i < HIDDENNO + 1; ++i) {
    wo[i] = drnd();
  }
}

// フィルタの初期化
void initfilter(double filter[FILTERNO][FILTERSIZE][FILTERSIZE]) {
  int i, j, k;

  for (i = 0; i < FILTERSIZE; ++i) {
    for (j = 0; j < FILTERSIZE; ++j) {
      for (k = 0; k < FILTERSIZE; ++k) {
        filter[i][j][k] = drnd();
      }
    }
  }
}


// 乱数の生成
double drnd() {
  double rndno;

  while((rndno = (double)rand() / RAND_MAX) == 1.0);
  rndno = rndno * 2 - 1;  // -1から1までの乱数を生成
  return rndno;
}


// プーリングの計算
void pool(double convout[][INPUTSIZE], double poolout[][POOLOUTSIZE]) {
  int i, j;

  for (i = 0; i < POOLOUTSIZE; ++i) {
    for (j = 0; j < POOLOUTSIZE; ++j) {
      poolout[i][j] = maxpooling(convout, i, j);
    }
  }
}


double maxpooling(double convout[][INPUTSIZE], int i, int j) {
  int m, n;
  double max;
  int halfpool = POOLSIZE / 2;

  max = convout[i * POOLOUTSIZE + 1 + halfpool][j * POOLOUTSIZE + 1 + halfpool];
  for (m = POOLOUTSIZE * i + 1; m <= POOLOUTSIZE * i + 1 + (POOLSIZE - halfpool); ++m) {
    for (n = POOLOUTSIZE * j + 1; n <= POOLOUTSIZE * j + 1 + (POOLSIZE - halfpool); ++n) {
      if (max < convout[m][n]) max = convout[m][n];
    }
  }

  return max;
}


void showdata(double e[][INPUTSIZE][INPUTSIZE], int t[], int n_of_e) {
  int i = 0, j = 0, k = 0;

  for (i = 0; i < n_of_e; ++i) {
    std::cout << "N=" << i << ", category=" << i << "\n";
    for (j = 0; j < INPUTSIZE; ++j) {
      for (k = 0; k < INPUTSIZE; ++k) {
        std::cout << e[i][j][k] << " ";
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  }
}

int getdata(double e[][INPUTSIZE][INPUTSIZE], int t[]) {
  int i = 0, j = 0, k = 0;

  while (scanf("%d", &t[i]) != EOF) {
    while (scanf("%lf", &e[i][j][k]) != EOF) {
	  ++k;
      if (k >= INPUTSIZE) {
        k = 0;
        ++j;
        if (j >= INPUTSIZE) break;
      }
    }
    j = 0;
    k = 0;
    ++i;
  }

  return i;
}


// 畳み込み計算
void conv(double filter[][FILTERSIZE], double e[][INPUTSIZE],
          double convout[][INPUTSIZE])
{
  int i = 0, j = 0;
  int startpoint = FILTERSIZE / 2;

  for (i = startpoint; i < INPUTSIZE - startpoint; ++i) {
    for (j = startpoint; j < INPUTSIZE - startpoint; ++j) {
      convout[i][j] = calcconv(filter, e, i, j);
    }
  }
}


double calcconv(double filter[][FILTERSIZE],
                double e[][INPUTSIZE], int i, int j)
{
  int m, n;
  double sum = 0;

  for (m = 0; m < FILTERSIZE; ++m) {
    for (n = 0; n < FILTERSIZE; ++n) {
      sum += e[i - FILTERSIZE / 2 + m][j - FILTERSIZE / 2 + n] * filter[m][n];
    }
  }

  return sum;
}

double f(double u) {
  return 1.0 / (1.0 + exp(-u));
}
