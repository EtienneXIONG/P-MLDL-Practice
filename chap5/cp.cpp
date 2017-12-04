/*
 * cp.cpp
 *
 * Purpose: 機械学習と深層学習, 畳み込みとプーリング
 * Created by: Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>
 */

#include <iostream>
#include <stdio.h>
#include <math.h>


const int INPUTSIZE = 11;   // 入力数
const int FILTERSIZE = 3;   // フィルタの大きさ
const int POOLSIZE = 3;     // プーリングのサイズ
const int POOLOUTSIZE = 3;  // プーリングの出力サイズ

void conv(double filter[][FILTERSIZE],
          double e[][INPUTSIZE],
          double convout[][INPUTSIZE]);  // 畳み込みの計算
double calcconv(double filter[][FILTERSIZE],
                double e[][INPUTSIZE], int i, int j);  // フィルタの適応
void convres(double convout[][INPUTSIZE]);  // 畳み込みの結果出力
void pool(double convout[][INPUTSIZE], double poolout[][POOLSIZE]);  // プーリングの計算
double maxpooling(double convout[][INPUTSIZE], int i, int j);  // 最大値プーリング
void poolres(double poolout[][POOLOUTSIZE]);  // 結果出力
void getdata(double e[][INPUTSIZE]);  // データ読み込み


int main()
{
  double filter[FILTERSIZE][FILTERSIZE] = {
    {0, 1, 0}, {0, 1, 0}, {0, 1, 0}  // 縦フィルタ
  };
  double e[INPUTSIZE][INPUTSIZE];  // 入力データ
  double convout[INPUTSIZE][INPUTSIZE] = { 0 };  // 畳み込み出力
  double poolout[POOLOUTSIZE][POOLOUTSIZE];      // 出力データ

  // 入力データの読み込み
  getdata(e);

  // 畳み込みの計算
  conv(filter, e, convout);

  // 畳み込みの出力
  convres(convout);

  // プーリングの計算
  pool(convout, poolout);

  // 結果の出力
  poolres(poolout);

  return 0;
}


// 結果出力
void poolres(double poolout[][POOLOUTSIZE]) {
  int i, j;

  for (i = 0; i < POOLSIZE; ++i) {
    for (j = 0; j < POOLSIZE; ++j) {
      std::cout << poolout[i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}


// プーリングの計算
void pool(double convout[][INPUTSIZE], double poolout[][POOLOUTSIZE]) {
  int i, j;

  for (i = 0; i < POOLSIZE; ++i) {
    for (j = 0; j < POOLSIZE; ++j) {
      poolout[i][j] = maxpooling(convout, i, j);
    }
  }
}


double maxpooling(double convout[][INPUTSIZE], int i, int j) {
  int m, n;
  double max;
  int halfpool = POOLSIZE / 2;

  max = convout[i * POOLOUTSIZE + 1 + halfpool][j * POOLSIZE + 1 + halfpool];
  for (m = POOLSIZE * i + 1; m <= POOLOUTSIZE * i + 1 + (POOLSIZE - halfpool); ++m) {
    for (n = POOLSIZE * j + 1; n <= POOLSIZE * j + 1 + (POOLSIZE - halfpool); ++n) {
      if (max < convout[m][n]) max = convout[m][n];
    }
  }

  return max;
}


void convres(double convout[][INPUTSIZE]) {
  int i, j;

  for (i = 1; i < INPUTSIZE - 1; ++i) {
    for (j = 1; j < INPUTSIZE - 1; ++j) {
      std::cout << convout[i][j] << " ";
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}


void getdata(double e[][INPUTSIZE]) {
  int i = 0, j = 0;

  while(scanf("%lf", &e[i][j]) != EOF) {
    ++j;
    if (j >= INPUTSIZE) {  // 次のデータ
      j = 0;
      ++i;
      if (i >= INPUTSIZE) break;
    }
  }
}


void conv(double filter[][FILTERSIZE],
          double e[][INPUTSIZE], double convout[][INPUTSIZE])
{
  int i = 0, j = 0;
  int startpoint = FILTERSIZE / 2;

  for (i = startpoint; i < INPUTSIZE - startpoint; ++i) {
    for (j = startpoint; j < INPUTSIZE - startpoint; ++j) {
      convout[i][j] = calcconv(filter, e, i, j);
    }
  }
}


double calcconv(double filter[][FILTERSIZE], double e[][INPUTSIZE], int i, int j) {
  int m, n;
  double sum = 0;

  for (m = 0; m < FILTERSIZE; ++m) {
    for (n = 0; n < FILTERSIZE; ++n) {
      sum += e[i - FILTERSIZE / 2 + m][j - FILTERSIZE / 2 + n] * filter[m][n];
    }
  }

  return sum;
}
