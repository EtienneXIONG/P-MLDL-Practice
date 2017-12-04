/*
 * learnstock.cpp
 *
 * Purpose: 機械学習と深層学習 chap2, パターンマッチング
 * Created by: Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>
 */

#include <stdio.h>
#include <iostream>


// 定数の定義
const int OK = 1;
const int NG = 0;

const int SETSIZE = 100;   // 学習データ・セットの大きさ
const int CNO = 10;        // 学習データの桁数 (10社分)
const int GENMAX = 10000;  // 解候補生成回数
const int SEED = 32767;    // 乱数のシード


// プロトタイプ宣言
void readdata(int data[SETSIZE][CNO], int teacher[SETSIZE]);
int rand012();
int calcscore(int data[SETSIZE][CNO], int teacher[SETSIZE], int answer[CNO]);


int main()
{
  int i, j;
  int score = 0;
  int answer[CNO];
  int data[SETSIZE][CNO];  // 学習データ・セット
  int teacher[SETSIZE];    // 教師データ
  int bestScore = 0;
  int bestanswer[CNO];     // 探索途中での最良解

  srand(SEED);  // 乱数の初期化
  readdata(data, teacher);  // 学習データ・セットの読み込み

  // 解候補生成と検査
  for (i = 0; i < GENMAX; ++i) {
    for (j = 0; j < CNO; ++j) {
      answer[j] = rand012();
    }

    // 検査
    score = calcscore(data, teacher, answer);

    // 最良スコアの更新
    if (score > bestScore) {
      for (j = 0; j < CNO; ++j) bestanswer[j] = answer[j];
      bestScore = score;

      // 表示
      for (j = 0; j < CNO; ++j) std::cout << bestanswer[j];
      std::cout << ": score = " << bestScore << "\n";
    }
  }

  // 再了解の出力
  std::cout << "bestScore" << "\n";
  for (j = 0; j < CNO; ++j) std::cout << bestanswer[j];
  std::cout << ": score = " << bestScore << "\n";

  return 0;
}


// 解スコアの計算
int calcscore(int data[SETSIZE][CNO], int teacher[SETSIZE], int answer[CNO]) {
  int score = 0;
  int point;
  int i, j;

  for (i = 0; i < SETSIZE; ++i) {
    point = 0;
    for (j = 0; j < CNO; ++j) {
      if (answer[j] == 2) ++point;  // ワイルドカード
      else if (answer[j] == data[i][j])  ++point;  // 一致
    }

    if ((point == CNO) && (teacher[i] == 1)) ++score;
    else if ((point != CNO) && (teacher[i] == 0)) ++score;
  }

  return score;
}

// 学習データ・セットの読み込み
void readdata(int data[SETSIZE][CNO], int teacher[SETSIZE]) {
  int i, j;

  for (i = 0; i < SETSIZE; ++i) {
    for (j = 0; j < CNO; ++j) {
      scanf("%d", &data[i][j]);
    }

    scanf("%d", &teacher[i]);
  }
}

// 乱数を生成
int rand012() {
  int rnd;

  while ((rnd = rand()) == RAND_MAX);
  return (double)rnd / RAND_MAX * 3;
}
