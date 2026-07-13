#pragma once
#include <vector>
#include <cmath>
#include "matrix.hpp"

using namespace std;

namespace StatMagic {

    // 极其暴力的 3x3 矩阵求逆（专门用来算西瓜协方差）
    inline Matrix inverse_3x3(Matrix M) {
        Matrix res(3, 3);
        auto& a = M.a;
        double det = a[0][0]*(a[1][1]*a[2][2] - a[1][2]*a[2][1])
                   - a[0][1]*(a[1][0]*a[2][2] - a[1][2]*a[2][0])
                   + a[0][2]*(a[1][0]*a[2][1] - a[1][1]*a[2][0]);
        
        // 绝对防御：防止行列式为 0（矩阵不可逆黑洞）
        if (abs(det) < 1e-9) det = 1e-9; 
        
        double invdet = 1.0 / det;
        res.a[0][0] = (a[1][1]*a[2][2] - a[2][1]*a[1][2]) * invdet;
        res.a[0][1] = (a[0][2]*a[2][1] - a[0][1]*a[2][2]) * invdet;
        res.a[0][2] = (a[0][1]*a[1][2] - a[0][2]*a[1][1]) * invdet;
        res.a[1][0] = (a[1][2]*a[2][0] - a[1][0]*a[2][2]) * invdet;
        res.a[1][1] = (a[0][0]*a[2][2] - a[0][2]*a[2][0]) * invdet;
        res.a[1][2] = (a[1][0]*a[0][2] - a[0][0]*a[1][2]) * invdet;
        res.a[2][0] = (a[1][0]*a[2][1] - a[2][0]*a[1][1]) * invdet;
        res.a[2][1] = (a[2][0]*a[0][1] - a[0][0]*a[2][1]) * invdet;
        res.a[2][2] = (a[0][0]*a[1][1] - a[1][0]*a[0][1]) * invdet;
        return res;
    }

    // 神级结构体：西瓜边界！
    struct Watermelon {
        int label;
        Matrix mu;      // 3x1 的均值向量 (西瓜的质心)
        Matrix cov_inv; // 3x3 的协方差逆矩阵 (西瓜的形状和量纲)
        
        Watermelon() {
            mu.set_size(3, 1);
            cov_inv.set_size(3, 3);
        }

        // 种西瓜（用训练集锁定高维分布边界）
        void grow(int l, const vector<Matrix>& X_list) {
            label = l;
            int n = X_list.size();
            if (n == 0) return;

            // 1. 算均值 mu (质心)
            mu.all_set(0);
            for (auto& x : X_list) mu = mu + x;
            mu = mu * (1.0 / n);

            // 2. 算协方差矩阵 cov (散布情况)
            Matrix cov(3, 3);
            cov.all_set(0);
            for (auto& x : X_list) {
                Matrix diff = x - mu;
                // 矩阵乘法：(3x1) * (1x3) = 3x3
                cov = cov + (diff * diff.transpose());
            }
            cov = cov * (1.0 / n);

            // 3. 极其关键的物理保护：在对角线加白噪声，防止单一特征完全一致导致矩阵坍缩！
            cov.a[0][0] += 1e-4;
            cov.a[1][1] += 1e-4;
            cov.a[2][2] += 1e-4;

            // 4. 求逆！引擎点火！
            cov_inv = inverse_3x3(cov);
        }

        // 吃西瓜（算马氏距离平方：D^2）
        double get_distance(const Matrix& X) const {
            Matrix diff = X - mu;
            // D^2 = (X - mu)^T * cov_inv * (X - mu)
            Matrix D2 = diff.transpose() * cov_inv * diff;
            return D2.a[0][0];
        }
    };
}