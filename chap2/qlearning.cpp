/*
 * qlearning.cpp
 *
 * Purpose: 機械学習と深層学習, chap2, qlearning
 * Created by: Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>
 */

#include <iostream>

const int GENMAX = 1000;    // 学習の繰り返し数
const int NODENO = 15;      // Q値のノード数
const double ALPHA = 0.1;   // 学習係数
const double GAMMA = 0.9;   // 割引率
const double EPSILON = 0.3; // 行動選択のランダム性を決定
const int SEED = 32767;     // 乱数のシード

int rand100();   // 0-100を返す
int rand01();    // 0 or 1を返す
double rand1();  // 0-1を返す
void printqvalue(int qvalue[NODENO]);    // Q値出力
int selecta(int s, int qvalue[NODENO]);  // 行動選択
int updateq(int s, int qvalue[NODENO]);  // Q値更新


int main()
{
  int i;
  int s;  // 状態
  int t;  // 時刻
  int qvalue[NODENO];  // Q値

  srand(SEED);

  // Q値の初期化
  std::cout << "--Initialize Q-value--" << "\n";
  for (i = 0; i < NODENO; ++i) qvalue[i] = rand100();
  printqvalue(qvalue);

  std::cout << "--Start Learning--" << "\n";

  // 学習
  for (i = 0; i < GENMAX; ++i) {
    s = 0;  // 行動選択の初期応対
    for (t = 0; t < 3; ++t) {  // 最下段まで繰り返す
      // 行動選択
      s = selecta(s, qvalue);
      // Q値の更新
      qvalue[s] = updateq(s, qvalue);
    }

    // Q値の出力
    printqvalue(qvalue);
  }

  return 0;
}


// Q値の更新
int updateq(int s, int qvalue[NODENO]) {
  int qv;    // 更新されるq値
  int qmax;  // q値の最大値

  if (s > 6) {  // 最下段の場合
    // 報酬の付与
    if (s == 14) qv = qvalue[s] + ALPHA * (1000 - qvalue[s]);

    // 報酬を与えるノードを増やす場合
    // else if (s == 11) qv = qvalue[s] + ALPHA * (500 - qvalue[s]);

    // 報酬なし
    else qv = qvalue[s];
  } else {  // 最下段以外
    if ((qvalue[2 * s + 1]) > (qvalue[2 * s + 2])) {
      qmax = qvalue[2 * s + 1];
    } else {
      qmax = qvalue[2 * s + 2];
    }

    qv = qvalue[s] + ALPHA * (GAMMA * qmax - qvalue[s]);
  }

  return qv;
}


// 行動を選択
int selecta(int olds, int qvalue[NODENO]) {
  int s;

  // ε-greedy法による行動選択
  if (rand1() < EPSILON) {  // ランダムに行動
    if (rand01() == 0 ) s = 2 * olds + 1;
    else s = 2 * olds + 2;
  } else {  // Q値最大値を選択
    if ((qvalue[2 * olds + 1]) > (qvalue[2 * olds + 2])) {
      s = 2 * olds + 1;
    } else {
      s = 2 * olds + 2;
    }
  }

  return s;
}


// q値を出力
void printqvalue(int qvalue[NODENO]) {
  int i;

  for (i = 0; i < NODENO; ++i) {
    std::cout << qvalue[i] << " ";
  }
  std::cout << "\n";
}


double rand1() {
  return (double)rand() / RAND_MAX;
}

int rand01() {
  int rnd;

  while ((rnd = rand()) == RAND_MAX);
  return (int)((double)rnd / RAND_MAX * 2);
}

int rand100() {
  int rnd;

  while ((rnd = rand()) == RAND_MAX);
  return (int)((double)rnd / RAND_MAX * 101);
}
