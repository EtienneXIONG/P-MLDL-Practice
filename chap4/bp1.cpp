/*
 * bp1.cpp
 *
 * Purpose: 機械学習と深層学習, chap4, バックプロパゲーション
 * Created by: Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>
 */

#include <iostream>
#include <stdio.h>
#include <math.h>


const int INPUTNO = 3;       // 入力層のセル数
const int HIDDENNO = 3;      // 中間層のセル数
const double ALPHA = 10;     // 学習係数
const int SEED = 65535;      // 乱数のシード
const int MAXINPUTNO = 100;  // 学習データの最大個数
const int BIGNUM = 100;      // 誤差の初期値
const double LIMIT = 0.001;  // 誤差の上限値

double f(double u);  // シグモイド関数
void initwh(double wh[HIDDENNO][INPUTNO + 1]);  // 中間層の重み初期化
void initwo(double wo[HIDDENNO + 1]);           // 出力層の重み初期化
double drnd();  // 乱数の生成
void print(double wh[HIDDENNO][INPUTNO + 1], double wo[HIDDENNO + 1]);  // 結果の出力
double forward(double wh[HIDDENNO][INPUTNO + 1],
               double wo[HIDDENNO + 1],
               double hi[],
               double  e[INPUTNO + 1]);  // 順方向の計算
void olearn(double wo[HIDDENNO + 1],
            double hi[],
            double  e[INPUTNO + 1],
            double  o);  // 出力層の重みの調整
int getdata(double e[][INPUTNO + 1]);  // 学習データ読み込み
void hlearn(double wh[HIDDENNO][INPUTNO + 1],
            double wo[HIDDENNO + 1],
            double hi[],
            double  e[INPUTNO + 1],
            double  o);  // 中間層の重みの調整


int main()
{
  double wh[HIDDENNO][INPUTNO + 1];    // 中間層の重み
  double wo[HIDDENNO + 1];             // 出力層の重み
  double  e[MAXINPUTNO][INPUTNO + 1];  // 学習データ・セット
  double hi[HIDDENNO + 1];             // 中間層の出力
  double o;             // 出力
  double err = BIGNUM;  // 誤差の評価
  int i, j;
  int n_of_e;           // 学習データの個数
  int count = 0;        // 繰り返し回数

  srand(SEED);

  // 重みの初期化
  initwh(wh);  // 中間層
  initwo(wo);  // 出力層
  print(wh, wo);  // 出力

  // 学習データの読み込み
  n_of_e = getdata(e);
  std::cout << "Number of data: " << n_of_e << "\n";

  // 学習
  while(err > LIMIT) {
    err = 0.0;
    for (j = 0; j < n_of_e; ++j) {
      // 順方向の計算
      o = forward(wh, wo, hi, e[j]);
      // 出力層の重みの調整
      olearn(wo, hi, e[j], o);
      // 中間層の重みの調整
      hlearn(wh, wo, hi, e[j], o);
      // 誤差の計算
      err += (o - e[j][INPUTNO]) * (o - e[j][INPUTNO]);
    }
    ++count;

    // 誤差の出力
    std::cout << count << ": " << err << "\n";
  }  // 学習終了

  // 結合荷重の出力
  print(wh, wo);

  // 学習データに対する出力
  for (i = 0; i < n_of_e; ++i) {
    std::cout << i << ": ";
    for (j = 0; j < INPUTNO + 1; ++j) std::cout << e[i][j] << " ";
    o = forward(wh, wo, hi, e[i]);
    std::cout << o << "\n";
  }

  return 0;
}


// 中間層の重み学習
void hlearn(double wh[HIDDENNO][INPUTNO + 1],
            double wo[HIDDENNO + 1],
            double hi[], double e[INPUTNO + 1], double o)
{
  int i, j;
  double dj;  // 中間層の重み計算に利用

  for (j = 0; j < HIDDENNO; ++j) {
    dj = hi[j] * (1 - hi[j]) * wo[j] * (e[INPUTNO] - o) * o * (1 - o);
    for (i = 0; i < INPUTNO; ++i) wh[j][i] += ALPHA * e[i] * dj;  // i番目の重みを処理
    wh[j][i] += ALPHA * (-1.0) * dj;  // 閾値の学習
  }
}

// 学習データの読み込み
int getdata(double e[][INPUTNO + 1]) {
  int n_of_e = 0;  // データ・セットの個数
  int j = 0;

  // データの入力
  while (scanf("%lf", &e[n_of_e][j]) != EOF) {
    ++j;
    if (j > INPUTNO) {  // 次のデータ
      j = 0;
      ++n_of_e;
    }
  }

  return n_of_e;
}


// 出力層の重み学習
void olearn(double wo[HIDDENNO + 1],
            double hi[], double e[INPUTNO + 1], double o)
{
  int i;
  double d;  // 重み計算に利用

  d = (e[INPUTNO] - o) * o * (1 - o);  // 誤差の計算
  for (i = 0; i < HIDDENNO; ++i) {
    wo[i] += ALPHA * hi[i] * d;  // 重みの学習
  }

  wo[i] += ALPHA * (-1.0) * d;  // 閾値の学習
}


// 順歩行の計算
double forward(double wh[HIDDENNO][INPUTNO + 1],
              double wo[HIDDENNO + 1], double hi[],
              double e[INPUTNO + 1])
{
  int i, j;
  double u;  // 重み付き和の計算
  double o;  // 出力の計算

  // hi の計算
  for (i = 0; i < HIDDENNO; ++i) {
    u = 0;
    for (j = 0; j < INPUTNO; ++j) u += e[j] * wh[i][j];
    u -= wh[i][j];  // 閾値の処理
    hi[i] = f(u);
  }

  // 出力oの計算
  o = 0;
  for (i = 0; i < HIDDENNO; ++i) o += hi[i] * wo[i];
  o -= wo[i];  // 閾値の処理

  return f(o);
}


// 結果の出力
void print(double wh[HIDDENNO][INPUTNO + 1], double wo[HIDDENNO + 1]) {
  int i, j;

  for (i = 0; i < HIDDENNO; ++i) {
    for (j = 0; j < INPUTNO + 1; ++j) std::cout << wh[i][j] << " ";
  }
  std::cout << "\n";

  for (i = 0; i < HIDDENNO + 1; ++i) std::cout << wo[i] << " ";
  std::cout << "\n";
}


// 中間層の重みの初期化
void initwh(double wh[HIDDENNO][INPUTNO + 1]) {
  int i, j;

  for (i = 0; i < HIDDENNO; ++i) {
    for (j = 0; j < INPUTNO + 1; ++j) {
      wh[i][j] = drnd();
    }
  }
}

// 出力層の重み初期化
void initwo(double wo[HIDDENNO + 1]) {
  int i;

  for (i = 0; i < HIDDENNO + 1; ++i) {
    wo[i] = drnd();
  }
}


// 乱数の生成
double drnd() {
  double rndno;

  while ((rndno = (double)rand() / RAND_MAX) == 0);
  rndno = rndno * 2 - 1;  // -1から1までの乱数を生成
  return rndno;
}


// シグモイド関数
double f(double u) {
  return 1.0 / (1.0 + exp(-u));
}
