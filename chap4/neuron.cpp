/*
 * neuron.cpp
 *
 * Purpose: 機械学習と深層学習, chap4, 単体のニューロン
 * Created by: Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>
 */

#include <stdio.h>
#include <iostream>
#include <math.h>


const int INPUTNO = 2;       // 入力数
const int MAXINPUTNO = 100;  // データの最大個数

double f(double u);  // 伝達関数
void initw(double w[INPUTNO + 1]);  // 重みと閾値の初期化
double forward(double w[INPUTNO + 1], double e[INPUTNO]);  // 順方向の計算
int getdata(double e[][INPUTNO]);  // データの読み込み

int main()
{
  double w[INPUTNO + 1];  // 重みと閾値
  double e[MAXINPUTNO][INPUTNO];  // データ・セット
  double o;  // 出力
  int i, j;
  int n_of_e;  // データの個数

  // 重みの初期化
  initw(w);

  // 入力データの読み込み
  n_of_e = getdata(e);
  std::cout << "Number of data: " << n_of_e << "\n";

  // 計算の本体
  for (i = 0; i < n_of_e; ++i) {
    std::cout << i << ": ";
    for (j = 0; j < INPUTNO; ++j) std::cout << e[i][j] << " ";
    o = forward(w, e[i]);
    std::cout << o << "\n";
  }

  return 0;
}


// 学習データの読み込み
int getdata(double e[][INPUTNO]) {
  int n_of_e = 0;  // データ・セットの個数
  int j = 0;

  while (scanf("%lf", &e[n_of_e][j]) != EOF) {
    ++j;
    if (j >= INPUTNO) {  // 次のデータ
      j = 0;
      ++n_of_e;
    }
  }

  return n_of_e;
}



double forward(double w[INPUTNO + 1], double e[INPUTNO]) {
  int i;
  double u, o;

  u = 0;
  for (i = 0; i < INPUTNO; ++i) u += e[i] * w[i];
  u -= w[i];  // 閾値の処理

  // 出力値の計算
  o = f(u);
  return o;
}



void initw(double w[INPUTNO + 1]) {
  // 定数を荷重として与える
  w[0] = 1;
  w[1] = 1;
  w[2] = 1.5;
}



double f(double u) {
  // ステップ関数の計算
  if (u >= 0) return 1.0;
  else return 0.0;
}
