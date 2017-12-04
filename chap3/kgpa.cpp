/*
 * kgpa.cpp
 *
 * Purpose: 機械学習と深層学習, chap3, GA
 * Created by: Keisuke Okumura <okumura.k@coord.c.titech.ac.jp>
 */

#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>


const int MAXVALUE = 100;  // 重さと価値の最大値
const int N = 30;  // 荷物の個数
const int WEIGHTLIMIT = (N * MAXVALUE / 4);  // 重量制限
const int POOLSIZE = 30;  // プールサイズ
const int LASTG = 100;  // 打ち切り世代
const double MRATE = 0.01;  // 突然変異の確立
const int SEED = 32767;  // 乱数のシード
const int YES = 1;
const int NO  = 0;


void initparcel();  // 荷物の初期化
int evalfit(int gene[]);  // 適応度の計算
void mating(int pool[POOLSIZE][N], int ngpool[POOLSIZE * 2][N]);  // 交叉
void mutation(int ngpool[POOLSIZE][N]);  // 突然変異
void printp(int pool[POOLSIZE][N]);  // 結果出力
void initpool(int pool[POOLSIZE][N]);  // 初期集団の生成
int rndn(int n);  // n未満の乱数の生成
int notval(int v);  // 真理値の反転
int selectp(int roulette[POOLSIZE], int totalfitness);  // 親の選択
void crossing(int m[], int p[], int c1[], int c2[]);  // 特定の2染色体の交叉
void selectng(int ngpool[POOLSIZE * 2][N], int pool[POOLSIZE][N]);  // 次世代の選択


int parcel[N][2];  // 荷物

int main(int argc, char *argv[])
{
  int pool[POOLSIZE][N];  // 染色体プール
  int ngpool[POOLSIZE * 2][N];  // 次世代染色体プール
  int generation;  // 現在の世代数

  srand(SEED);

  initparcel();  // 荷物の初期化

  initpool(pool);  // 初期集団の形成

  // 学習
  for (generation = 0; generation < LASTG; ++generation) {
    std::cout << "Generation: " << generation << "\n";
    std::cout << "Mating" << "\n";
    mating(pool, ngpool);  // 交叉
    std::cout << "Mutation" << "\n";
    mutation(ngpool);      // 突然変異
    std::cout << "Selecting" << "\n";
    selectng(ngpool, pool);  // 次世代の選択
    printp(pool);  // 結果出力
    std::cout << "--------------------\n";
  }

  return 0;
}


void initparcel() {
  int i = 0;
  while((i < N)
        && scanf("%d %d", &parcel[i][0], &parcel[i][1]) != EOF) {
    ++i;
  }
}



void selectng(int ngpool[POOLSIZE*2][N], int pool[POOLSIZE][N]) {
  int i, j, c;
  int totalfitness = 0;  // 適応度の合計値
  int roulette[POOLSIZE * 2];  // 適応度を格納
  int ball;  // 選択位置の数値
  int acc = 0;  // 適応度の積算値

  for (i = 0; i < POOLSIZE; ++i) {
    // ルーレットの作成
    totalfitness = 0;
    for (c = 0; c < POOLSIZE * 2; ++c) {
      roulette[c] = evalfit(ngpool[c]);
      totalfitness += roulette[c];
    }

    // 染色体を一つ選ぶ
    ball = rndn(totalfitness);
    acc = 0;
    for (c = 0; c < POOLSIZE * 2; ++c) {
      acc += roulette[c];  // 適応度を積算
      if (acc > ball) break;  // 対応する遺伝子
    }

    // 染色体のコピー
    for (j = 0; j < N; ++j) {
      pool[i][j] = ngpool[c][j];
    }
  }
}


// 親を選択
int selectp(int roulette[POOLSIZE], int totalfitness) {
  int i;
  int ball;
  int acc = 0;

  ball = rndn(totalfitness);
  for (i = 0; i < POOLSIZE; ++i) {
    acc += roulette[i];
    if (acc > ball) break;
  }

  return i;
}


void mating(int pool[POOLSIZE][N], int ngpool[POOLSIZE * 2][N]) {
  int i;
  int totalfitness = 0;
  int roulette[POOLSIZE];
  int mama, papa;

  // ルーレットの作成
  for (i = 0; i < POOLSIZE; ++i) {
    roulette[i] = evalfit(pool[i]);
    totalfitness += roulette[i];
  }

  // 選択と交叉を繰り返す
  for (i = 0; i < POOLSIZE; ++i) {
    do {  // 親の選択
      mama = selectp(roulette, totalfitness);
      papa = selectp(roulette, totalfitness);
    } while (mama == papa);  // 重複の削除

    crossing(pool[mama], pool[papa], ngpool[i * 2], ngpool[i * 2 + 1]);
  }
}

// 交叉
void crossing(int m[], int p[], int c1[], int c2[]) {
  int j;
  int cp;

  cp = rndn(N);  // 交叉点の決定

  // 前半部分のコピー
  for (j = 0; j < cp; ++j) {
    c1[j] = m[j];
    c2[j] = p[j];
  }

  // 後半部分のコピー
  for (; j < N; ++j) {
    c2[j] = m[j];
    c1[j] = p[j];
  }
}


int evalfit(int g[]) {
  int pos;  // 遺伝子座の指定
  int value = 0;  // 評価値
  int weight = 0;  // 重量

  // 各遺伝子座を調べて重量と評価値を計算
  for (pos = 0; pos < N; ++pos) {
    weight += parcel[pos][0] * g[pos];
    value += parcel[pos][1] * g[pos];
  }

  // 致死遺伝子の処理
  if (weight >= WEIGHTLIMIT) value = 0;
  return value;
}


void printp(int pool[POOLSIZE][N]) {
  int i, j;
  int fitness;  // 適応度
  double totalfitness = 0;
  int elite, bestfit = 0;  // エリート遺伝子の処理用

  for (i = 0; i < POOLSIZE; ++i) {
    std::cout << i << " : ";
    for (j = 0; j < N; ++j) std::cout << pool[i][j];
    fitness = evalfit(pool[i]);
    std::cout << " : fitness -> " << fitness << "\n";

    // エリート解
    if (fitness > bestfit) {
      bestfit = fitness;
      elite = i;
    }

    totalfitness += fitness;
  }

  // エリート解の適応度を出力
  std::cout << "Elite: " << elite << " -> " << bestfit << "\n";
  // 平均の適応度
  std::cout << "Mean -> " << totalfitness / POOLSIZE << "\n";
}


void initpool(int pool[POOLSIZE][N]) {
  int i, j;
  for (i = 0; i < POOLSIZE; ++i) {
    for (j = 0; j < N; ++j) {
      pool[i][j] = rndn(2);
    }
  }
}


int rndn(int l) {
  int rndno;
  while((rndno = ((double)rand() / RAND_MAX) * l) == l);
  return rndno;
}


void mutation(int ngpool[POOLSIZE * 2][N]) {
  int i, j;

  for (i = 0; i < POOLSIZE * 2; ++i) {
    for (j = 0; j < N; ++j) {
      if ((double)rndn(100) / 100.0 <= MRATE) {
        ngpool[i][j] = notval(ngpool[i][j]);
      }
    }
  }
}


int notval(int v) {
  if (v == YES) return NO;
  else return YES;
}
