#pragma once
#include <vector>
#include <complex>
#include <cmath>
#include "data_loader.hpp" // 让它认识 Image 结构体

namespace MathMagic {
    using namespace std;
    using Complex = complex<double>;
    const double PI = acos(-1.0);

    // 加上 inline 防止多文件包含时报错
    inline void fft_1d(vector<Complex>& a, bool invert) {
        int n = a.size();
        for (int i = 1, j = 0; i < n; i++) {
            int bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) swap(a[i], a[j]);
        }
        for (int len = 2; len <= n; len <<= 1) {
            double ang = 2 * PI / len * (invert ? -1 : 1);
            Complex wlen(cos(ang), sin(ang));
            for (int i = 0; i < n; i += len) {
                Complex w(1);
                for (int j = 0; j < len / 2; j++) {
                    Complex u = a[i + j], v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (invert) {
            for (Complex & x : a) x /= n;
        }
    }

    // 2D 傅里叶低通滤波器
    inline void apply_fft_filter(Image& img, int radius) {
        int N = 32; 
        vector<vector<Complex>> grid(N, vector<Complex>(N, 0));

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                double val = img.pixels[i][j];
                if ((i + j) % 2 == 1) val = -val; 
                grid[i][j] = Complex(val, 0);
            }
        }

        for (int i = 0; i < N; i++) fft_1d(grid[i], false);
        for (int j = 0; j < N; j++) {
            vector<Complex> col(N);
            for (int i = 0; i < N; i++) col[i] = grid[i][j];
            fft_1d(col, false);
            for (int i = 0; i < N; i++) grid[i][j] = col[i];
        }

        int cx = N / 2, cy = N / 2;
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) {
                int dist2 = (i - cx) * (i - cx) + (j - cy) * (j - cy);
                if (dist2 > radius * radius) {
                    grid[i][j] = Complex(0, 0); 
                }
            }
        }

        for (int i = 0; i < N; i++) fft_1d(grid[i], true);
        for (int j = 0; j < N; j++) {
            vector<Complex> col(N);
            for (int i = 0; i < N; i++) col[i] = grid[i][j];
            fft_1d(col, true);
            for (int i = 0; i < N; i++) grid[i][j] = col[i];
        }

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                double val = grid[i][j].real();
                if ((i + j) % 2 == 1) val = -val;
                
                int final_val = round(val);
                if (final_val < 0) final_val = 0;
                if (final_val > 255) final_val = 255;
                img.pixels[i][j] = final_val;
            }
        }
    }

    // 变分预处理：ROF 全变分去噪模型 (Total Variation Denoising)
    // lambda: 保真度旋钮 (越小越平滑，越大越忠于原图。推荐 0.02 ~ 0.08)
    // iters: 梯度下降迭代次数 (20~50次足够收敛)
    inline void apply_tv_denoising(Image& img, double lambda = 0.05, int iters = 30) {
        double u[28][28], f[28][28];
        
        // 1. 物理克隆：f 是永远不变的原始参照物，u 是我们要雕刻的艺术品
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                u[i][j] = f[i][j] = img.pixels[i][j];
            }
        }

        double dt = 0.2; // 时间步长，绝对不能大于 0.25，否则系统会物理爆炸（数值发散）
        double ep = 1e-5; // 防止除以零的黑洞

        // 2. 梯度下降迭代引擎启动
        for (int step = 0; step < iters; step++) {
            double unew[28][28];
            for (int i = 0; i < 28; i++) {
                for (int j = 0; j < 28; j++) {
                    // 获取上下左右邻居 (撞墙保护，边缘复用)
                    double u_up = i > 0 ? u[i - 1][j] : u[i][j];
                    double u_dn = i < 27 ? u[i + 1][j] : u[i][j];
                    double u_lt = j > 0 ? u[i][j - 1] : u[i][j];
                    double u_rt = j < 27 ? u[i][j + 1] : u[i][j];

                    // 计算偏导数差分
                    double ux_plus  = u_dn - u[i][j];
                    double ux_minus = u[i][j] - u_up;
                    double uy_plus  = u_rt - u[i][j];
                    double uy_minus = u[i][j] - u_lt;

                    // 计算非线性扩散系数 (梯度的倒数：越平坦的地方扩散越快，边缘处扩散停止！)
                    double c_dn = 1.0 / sqrt(ux_plus * ux_plus + ep);
                    double c_up = 1.0 / sqrt(ux_minus * ux_minus + ep);
                    double c_rt = 1.0 / sqrt(uy_plus * uy_plus + ep);
                    double c_lt = 1.0 / sqrt(uy_minus * uy_minus + ep);

                    // 计算散度 (Divergence)
                    double div = c_dn * ux_plus - c_up * ux_minus + c_rt * uy_plus - c_lt * uy_minus;

                    // 核心公式：梯度下降更新
                    unew[i][j] = u[i][j] + dt * (div - lambda * (u[i][j] - f[i][j]));
                }
            }
            
            // 状态同步更新
            for (int i = 0; i < 28; i++)
                for (int j = 0; j < 28; j++)
                    u[i][j] = unew[i][j];
        }

        // 3. 完美出炉，覆写回原机甲阵列
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                int val = round(u[i][j]);
                if (val < 0) val = 0;
                if (val > 255) val = 255;
                img.pixels[i][j] = val;
            }
        }
    }
}