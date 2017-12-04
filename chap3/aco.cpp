/*
 * aco.cpp
 *
 * Purpose: 機械学習と深層学習, chap3, Ant Colony Optimization
 * Created by: Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>
 */

#include <iostream>
#include <cmath>
#include <random>


const int NOA = 1000;    // アリの個体数, 増やさないと最適解にならない
const int ILIMIT = 2;    // 繰り返し数
const double Q = 3;      // フェロモン更新の定数
const double RHO = 0.8;  // 蒸発の定数
const int STEP = 10;     // 道のりのステップ数
const double EPSILON = 0.15;  // 行動選択のランダム性


void printp(double pheromene[2][STEP]);  // 表示
void printmstep(int mstep[NOA][STEP]);   // アリの行動
void walk(int cost[2][STEP], double pheromene[2][STEP], int (&mstep)[NOA][STEP]);
void update(int cost[2][STEP], double (&pheromene)[2][STEP], int mstep[NOA][STEP]);
double rand1();  // 0-1を返す
int rand01();    // 0 or 1を返す


int main()
{
  // 各ステップのコスト
  int cost[2][STEP] = {
    {1, 5, 1, 5, 1, 5, 1, 5, 1, 5},
    {5, 1, 5, 1, 5, 1, 5, 1, 5, 1}
  };

  // 各ステップのフェロモン量
  double pheromene[2][STEP] = { 0 };

  // 蟻が歩いた過程
  int mstep[NOA][STEP] = { 0 };

  int i;

  // 最適化
  for (i = 0; i < ILIMIT; ++i) {
    // フェロモンの状態を出力
    std::cout << i << ":\n";
    printp(pheromene);

    // アリを歩かせる
    walk(cost, pheromene, mstep);

    // フェロモンの更新
    update(cost, pheromene, mstep);
  }

  // フェロモンの状態出力
  std::cout << i << ":\n";
  printp(pheromene);

  return 0;
}


void update(int cost[2][STEP], double (&pheromene)[2][STEP], int mstep[NOA][STEP]) {
  int m;   // アリの個体番号
  int lm;  // アリの歩いた距離
  int i, j;
  double sum_lm = 0;  // アリの歩いた合計距離

  // フェロモンの蒸発
  for (i = 0; i < 2; ++i) {
    for (j = 0; j < STEP; ++j) {
      pheromene[i][j] *= RHO;
    }
  }

  // アリによる上塗り
  for (m = 0; m < NOA; ++m) {
    lm = 0;
    for (i = 0; i < STEP; ++i) lm += cost[mstep[m][i]][i];

    // フェロモンの上塗り
    for (i = 0; i < STEP; ++i) {
      pheromene[mstep[m][i]][i] += Q * (1.0 / lm);
    }
    sum_lm += lm;
  }

  std::cout << sum_lm / NOA << "\n";
}


void walk(int cost[2][STEP], double pheromene[2][STEP], int (&mstep)[NOA][STEP]) {
  int m;  // アリの個体番号
  int s;  // アリの現在ステップ位置

  for (m = 0; m < NOA; ++m) {
    for (s = 0; s < STEP; ++s) {
      // ε-greedy法による行動選択
      if ((rand1() < EPSILON)
          || (std::abs(pheromene[0][s] - pheromene[1][s]) < 1e-9)) {
        // ランダムに行動
        mstep[m][s] = rand01();
      } else {
        if (pheromene[0][s] > pheromene[1][s]) {
          mstep[m][s] = 0;
        } else {
          mstep[m][s] = 1;
        }
      }
    }
  }

  printmstep(mstep);
}


void printmstep(int mstep[NOA][STEP]) {
  int i, j;

  std::cout << "*mstep" << "\n";
  for (i = 0; i < NOA; ++i) {
    for (j = 0; j < STEP; ++j) std::cout << mstep[i][j] << " ";
    std::cout << "\n";
  }
}


void printp(double pheromene[2][STEP]) {
  int i, j;

  for (i = 0; i < 2; ++i) {
    for (j = 0; j < STEP; ++j) std::cout << pheromene[i][j] << " ";
    std::cout << "\n";
  }
}


double rand1() {
  std::random_device rnd;     // 非決定的な乱数生成器を生成
  std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
  std::uniform_real_distribution<double> zero2one(0, 1);
  return zero2one(mt);
}


int rand01() {
  std::random_device rnd;     // 非決定的な乱数生成器を生成
  std::mt19937 mt(rnd());     //  メルセンヌ・ツイスタの32ビット版、引数は初期シード値
  std::uniform_int_distribution<int> zeroORone(0, 1);        // [0, 1] 範囲の一様乱数
  return zeroORone(mt);
}
