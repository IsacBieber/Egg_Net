#include <bits/stdc++.h>
#include <iomanip>
#include "model_io.hpp"
#include "data_loader.hpp"
#include "math_magic.hpp"
#include "matrix.hpp"
#include "vision_features.hpp"
#include "egg_net.hpp"
#include "watermelon.hpp" // 引入你的统计学神兵！

using namespace std;

int main(void) 
{
    system("chcp 65001 > nul");
    cout << "[*] 终极 EGG_NET 深度学习机甲 + 西瓜引擎 联合启动 ..." << endl;

    int train_size = 6000;
    int test_size = 4000;
    int total_size = train_size + test_size;

    vector<Image> dataset = load_mnist_csv("mnist_train.csv", total_size); 
    cout << "[+] 成功装载 " << dataset.size() << " 张图片！火力准备就绪！" << endl;
    
    // ==========================================
    // 阶段一：西瓜农场培育 (只使用训练集)
    // ==========================================
    cout << "[*] 正在扫描前 " << train_size << " 张图的 3D 核心几何特征，准备种西瓜..." << endl;
    vector<vector<Matrix>> features_by_label(10);
    
    for (int i = 0; i < train_size; i++) {
        // 第一遍扫描，先把底层去噪做了
        MathMagic::apply_fft_filter(dataset[i], 12); 
        MathMagic::apply_tv_denoising(dataset[i], 0.05, 30);
        
        Matrix X_core(3, 1);
        X_core.a[0][0] = VisionFeatures::count_components(dataset[i]);
        X_core.a[1][0] = VisionFeatures::get_quotient(dataset[i]);           
        X_core.a[2][0] = VisionFeatures::curve_enegy(dataset[i]);     
        
        features_by_label[dataset[i].label].push_back(X_core);
    }

    cout << "[*] 正在计算 3D 协方差与马氏距离边界 ..." << endl;
    vector<StatMagic::Watermelon> farm(10);
    for (int k = 0; k < 10; k++) {
        farm[k].grow(k, features_by_label[k]);
    }
    cout << "[+] 10 个高维西瓜引力场部署完毕！" << endl;

    // ==========================================
    // 阶段二：萃取 794 维终极融合特征
    // ==========================================
    vector<Matrix> X_train, Y_train;
    cout << "[*] 正在萃取 794 维终极神级特征 (像素 + 西瓜概率) ..." << endl;
    
    for (int i = 0; i < train_size; i++) {
        Matrix X(794, 1); 
        int idx = 0;
        
        // 1. 装填 784 维原生像素
        for(int r = 0; r < 28; r++) {
            for(int c = 0; c < 28; c++) {
                X.a[idx++][0] = dataset[i].pixels[r][c] / 255.0; 
            }
        }
        
        // 2. 重新提取 3D 核心特征
        Matrix X_core(3, 1);
        X_core.a[0][0] = VisionFeatures::count_components(dataset[i]);
        X_core.a[1][0] = VisionFeatures::get_quotient(dataset[i]);           
        X_core.a[2][0] = VisionFeatures::curve_enegy(dataset[i]);
        
        // 3. 终极融合：注入 10 个西瓜的引力评分！
        for(int k = 0; k < 10; k++) {
            double d2 = farm[k].get_distance(X_core);
            // 魔法公式：将距离的平方用高斯核压缩到 0~1 的概率区间
            X.a[idx++][0] = exp(-d2 / 2.0); 
        }
        
        X_train.push_back(X);

        Matrix Y(10, 1);
        for(int k = 0; k < 10; k++) Y.a[k][0] = (k == dataset[i].label) ? 1.0 : 0.0;
        Y_train.push_back(Y);
    }

    // ==========================================
    // 阶段三：Egg_Net 点火训练 (注意输入维度变 794，并开启 CE 模式)
    // ==========================================
    Egg_Net egg({794, 64, 10}, "CE");
    bool has_memory = ModelIO::load_model(egg, "egg_net_v1.txt");

    if (!has_memory) {
        cout << "\n[*] EGG_NET 引擎点火！开始 100 轮交叉熵梯度狂飙！" << endl;
        int epochs = 100; 
        
        for (int epoch = 1; epoch <= epochs; epoch++) {
            db total_loss = 0;
            int correct = 0;
            
            for (int i = 0; i < X_train.size(); i++) {
                Matrix Pred = egg.forward(X_train[i]);
                
                // 原地洗牌计算 CE 损失 (注意 get_loss_ce 是昨天让你在 egg_net.hpp 加的)
                total_loss += egg.loss_type == "CE" ? get_loss_ce(Pred, Y_train[i]) : get_loss_mse(Pred - Y_train[i]);
                
                int best_guess = 0;
                db max_p = Pred.a[0][0];
                for(int k = 1; k < 10; k++) {
                    if(Pred.a[k][0] > max_p) {
                        max_p = Pred.a[k][0];
                        best_guess = k;
                    }
                }
                if (best_guess == dataset[i].label) correct++;

                Matrix G = Pred - Y_train[i];
                egg.backward(G); 
            }
            
            if (epoch % 5 == 0 || epoch == 1) {
                cout << "=> Epoch [" << epoch << "/" << epochs << "] | "
                    << "网络混沌度: " << fixed << setprecision(4) << total_loss / X_train.size() << " | "
                    << "训练集统治率: " << fixed << setprecision(2) << (correct * 100.0 / X_train.size()) << " %" << endl;
            }
        }
        ModelIO::save_model(egg, "egg_net_v1.txt");
    }

    // ==========================================
    // 阶段四：期末大考 (将西瓜制裁应用到测试集)
    // ==========================================
    cout << "\n[*] 正在准备期末考试卷... 提取测试集 794 维神级特征..." << endl;
    int test_correct = 0;
    
    for (int i = train_size; i < total_size; i ++) {
        MathMagic::apply_fft_filter(dataset[i], 12); 
        MathMagic::apply_tv_denoising(dataset[i], 0.05, 30);
        
        Matrix X_test(794, 1); 
        int idx = 0;
        for(int r = 0; r < 28; r++) {
            for(int c = 0; c < 28; c++) {
                X_test.a[idx++][0] = dataset[i].pixels[r][c] / 255.0;
            }
        }
        
        // 核心 3D 特征提取
        Matrix X_core(3, 1);
        X_core.a[0][0] = VisionFeatures::count_components(dataset[i]);
        X_core.a[1][0] = VisionFeatures::get_quotient(dataset[i]);           
        X_core.a[2][0] = VisionFeatures::curve_enegy(dataset[i]);
        
        // 感受 10 个西瓜在考场上的引力
        for(int k = 0; k < 10; k++) {
            double d2 = farm[k].get_distance(X_core);
            X_test.a[idx++][0] = exp(-d2 / 2.0); 
        }
        
        Matrix Pred = egg.forward(X_test);
        
        int best_guess = 0;
        db max_p = Pred.a[0][0];
        for(int k = 1; k < 10; k++) {
            if(Pred.a[k][0] > max_p) {
                max_p = Pred.a[k][0];
                best_guess = k;
            }
        }
        if (best_guess == dataset[i].label) test_correct++;
    }

    cout << "======================================" << endl;
    cout << "【最终 " << test_size << " 张大考成绩报告】" << endl;
    cout << "测试样本量: " << test_size << " 张 (网络绝对没见过的数据)" << endl;
    cout << "盲猜成功数: " << test_correct << " 张" << endl;
    cout << "真实泛化准确率: " << fixed << setprecision(2) << (test_correct * 100.0 / test_size) << " %" << endl;
    cout << "======================================" << endl;

    return 0;
}