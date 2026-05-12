#pragma once
#include <vector>
#include <cmath>
#include <random>
#include <chrono>
#include "matrix.hpp"

using namespace std;
using db = double;

// 解决多次 include 导致的随机数引擎重定义问题
inline mt19937_64 rnd(chrono::steady_clock::now().time_since_epoch().count());
inline uniform_real_distribution<double> dist_rnd(-1.0, 1.0);

struct Layer {
    db learning_rate;
    int W_row, B_row;
    bool linear;
    Matrix W, B, A;
    
    Layer() {}
    Layer(int W_row, int B_row, bool linear) : W_row(W_row), B_row(B_row), linear(linear) {
        W.set_size(W_row, B_row);
        B.set_size(W_row, 1);
        if (linear) {
            db limit = sqrt(6.0 / (W_row + B_row)); 
            for(int i = 0; i < W_row; i ++)
                for(int j = 0; j < B_row; j ++)
                    W.a[i][j] = dist_rnd(rnd) * limit;
        }
    }

    void set_learning_rate(db v) { learning_rate = v; }

    Matrix process(const Matrix& X) {
        Matrix res = A = X;
        if (linear) { 
            res = W * res + B;
        } else { // RELU
            for (int i = 0; i < res.row; i ++) {
                res.a[i][0] = max(res.a[i][0], 0.0);
            }
        }
        return res;
    }

    Matrix adjust(const Matrix& G) {
        Matrix new_G = G;
        if (linear) {
            new_G = W.transpose() * G;
            B = B - G * learning_rate; 
            W = W - (G * A.transpose()) * learning_rate;
        } else {
            for (int i = 0; i < G.row; i ++) {
                new_G.a[i][0] = A.a[i][0] > 0 ? G.a[i][0] : 0;
            }
        }
        return new_G;
    }
};

struct Egg_Net {
    int depth;
    vector <Layer> l;
    string loss_type;

    Egg_Net (const vector <int> &dims, string loss_type = "MSE") : loss_type(loss_type) {
        depth = (dims.size() - 1) * 2 - 1; 
        l.resize(depth);
        int idx = 0;
        for (int i = 0; i < dims.size() - 1; i ++) {
            l[idx] = Layer(dims[i + 1], dims[i], true); 
            l[idx].set_learning_rate(0.01); 
            idx ++;
            if (i < dims.size() - 2) {
                l[idx] = Layer(dims[i + 1], dims[i + 1], false); 
                l[idx].set_learning_rate(0.01);
                idx ++;
            }
        }
    }

    Matrix forward(Matrix X) {
        for (int i = 0; i < depth; i ++) {
            X = l[i].process(X);
        }
        return X;
    }

    void backward(Matrix G) {
        for (int i = depth - 1; i >= 0; i --) {
            G = l[i].adjust(G);
        }
    }
};

// 经典均方误差 (MSE)
inline db get_loss_mse(const Matrix& diff) {
    db res = 0;
    for(int i = 0; i < diff.row; i++) res += diff.a[i][0] * diff.a[i][0];
    return res * 0.5;
}

// 交叉熵 (CE) + 防爆 Softmax 归一化
// 注意参数 Pred 前面没有 const，我们要原地将得分洗成概率！
inline db get_loss_ce(Matrix& Pred, const Matrix& Y) {
    // 1. 寻找最大得分 (防爆处理，防止 e^x 产生 INF)
    db max_val = Pred.a[0][0];
    for (int i = 1; i < Pred.row; i++) {
        if (Pred.a[i][0] > max_val) max_val = Pred.a[i][0];
    }

    // 2. 原地计算指数并求和
    db sum_exp = 0;
    for (int i = 0; i < Pred.row; i++) {
        Pred.a[i][0] = exp(Pred.a[i][0] - max_val);
        sum_exp += Pred.a[i][0];
    }

    // 3. 计算交叉熵损失，并原地覆写为概率 P
    db loss = 0;
    for (int i = 0; i < Pred.row; i++) {
        Pred.a[i][0] /= sum_exp; // 这一步，得分变成了真正的概率分布
        
        // 计算 -Y * log(P)，加入 1e-8 防止 log(0) 黑洞
        loss -= Y.a[i][0] * log(Pred.a[i][0] + 1e-8); 
    }
    return loss;
}