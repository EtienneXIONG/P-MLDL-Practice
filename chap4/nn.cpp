/*
 * nn.cpp
 *
 * Purpose: 機械学習と深層学習, chap4, ニューラルネットの計算
 * Created by: Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>
 */

#include <iostream>
#include <stdio.h>
#include <math.h>

const int INPUTNO = 2;   // 入力層のセル数
const int HIDDENNO = 2;  // 中間層のセル数
const int MAXINPUTNO = 100;  // データの最大個数


double f(double u);  // 伝達関数
void initwh(double wh[HIDDENNO][INPUTNO + 1]);  // 中間層の重みの初期化
void initwo(double wo[HIDDENNO + 1]);           // 出力層の重みの初期化
double forward(double wh[HIDDENNO][INPUTNO + 1],
               double wo[HIDDENNO + 1],
               double hi[],
               double e[INPUTNO]);
int getdata(double e[][INPUTNO]);  // データ読み込み


int main()
{
  double wh[HIDDENNO][INPUTNO + 1];  // 中間層の重み
  double wo[HIDDENNO + 1];           // 出力層の重み
  double e[MAXINPUTNO][INPUTNO];     // データ・セット
  double hi[HIDDENNO + 1];           // 中間層の出力
  double o;  // 出力
  int i, j;
  int n_of_e;  // データの個数

  // 重みの初期化
  initwh(wh);
  initwo(wo);

  // 入力データの読み込み
  n_of_e = getdata(e);
  std::cout << "Number of data: " << n_of_e << "\n";

  // 計算の本体
  for (i = 0; i < n_of_e; ++i) {
    std::cout << i << ": ";

    for (j = 0; j < INPUTNO; ++j) std::cout << e[i][j] << " ";
    o = forward(wh, wo, hi, e[i]);
    std::cout << o << "\n";
  }

  return 0;
}

int getdata(double e[][INPUTNO]) {
  int n_of_e = 0;
  int j = 0;

  while(scanf("%lf", &e[n_of_e][j]) != EOF) {
    ++j;
    if (j >= INPUTNO) {
      j = 0;
      ++n_of_e;
    }
  }

  return n_of_e;
}


double forward(double wh[HIDDENNO][INPUTNO + 1],
               double wo[HIDDENNO + 1],
               double hi[],
               double e[INPUTNO]) {
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

  // 出力 o の計算
  o = 0;
  for (i = 0; i < HIDDENNO; ++i) o+= hi[i] * wo[i];

  o -= wo[i];

  return f(o);
}


// 中間層の重み初期化
void initwh(double wh[HIDDENNO][INPUTNO + 1]) {
  // 荷重を定数として与える
  wh[0][0] = -2;
  wh[0][1] = 3;
  wh[0][2] = -1;
  wh[1][0] = -2;
  wh[1][1] = 1;
  wh[1][2] = 0.5;
}

// 出力層の重み初期化
void initwo(double wo[HIDDENNO + 1]) {
  wo[0] = -60;
  wo[1] = 94;
  wo[2] = -1;
}

double f(double u) {
  // ステップ関数
  if (u >= 0) return 1.0;
  else return 0.0;
}
