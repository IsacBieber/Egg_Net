#include <bits/stdc++.h>
#include <iomanip>
#include "model_io.hpp"
#include "data_loader.hpp"
#include "math_magic.hpp"
#include "matrix.hpp"
#include "vision_features.hpp"
#include "egg_net.hpp"

using namespace std;

int main(void) 
{
    system("chcp 65001 > nul");
    cout << "[*] 终极 EGG_NET 深度学习机甲启动，核反应堆全开 ..." << endl;

    int train_size = 6000;
    int test_size = 4000;
    int total_size = train_size + test_size;

    vector<Image> dataset = load_mnist_csv("mnist_train.csv", total_size); 
    cout << "[+] 成功装载 " << dataset.size() << " 张图片！火力准备就绪！" << endl;
    
    vector<Matrix> X_train, Y_train;
    
    cout << "[*] 正在萃取前 " << train_size << " 张图的 787 维融合特征 (算力狂飙中，请稍候...) ..." << endl;
    for (int i = 0; i < train_size; i++) {
        MathMagic::apply_fft_filter(dataset[i], 12); 
        MathMagic::apply_tv_denoising(dataset[i], 0.05, 30);
        
        Matrix X(787, 1); 
        int idx = 0;
        for(int r = 0; r < 28; r++) {
            for(int c = 0; c < 28; c++) {
                X.a[idx++][0] = dataset[i].pixels[r][c] / 255.0; 
            }
        }
        X.a[idx++][0] = VisionFeatures::count_components(dataset[i]) / 3.0; 
        X.a[idx++][0] = VisionFeatures::get_quotient(dataset[i]);           
        X.a[idx++][0] = VisionFeatures::curve_enegy(dataset[i]) / 10.0;     
        
        X_train.push_back(X);

        Matrix Y(10, 1);
        for(int k = 0; k < 10; k++) Y.a[k][0] = (k == dataset[i].label) ? 1.0 : 0.0;
        Y_train.push_back(Y);
    }

    Egg_Net egg({787, 64, 10});

    // 【新增】：尝试开机加载记忆！
    bool has_memory = ModelIO::load_model(egg, "egg_net_v1.txt");

    // 如果没有记忆，才去进行那地狱般的 100 轮训练！
    if (!has_memory) {
        cout << "\n[*] EGG_NET 引擎点火！开始 100 轮地狱梯度下降！" << endl;
        int epochs = 100; 
        
        for (int epoch = 1; epoch <= epochs; epoch++) {
            db total_loss = 0;
            int correct = 0;
            
            for (int i = 0; i < X_train.size(); i++) {
                Matrix Pred = egg.forward(X_train[i]);
                
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
                total_loss += get_mse_loss(G);
                egg.backward(G); 
            }
            
            if (epoch % 5 == 0 || epoch == 1) {
                cout << "=> Epoch [" << epoch << "/" << epochs << "] | "
                    << "网络混沌度(Loss): " << fixed << setprecision(4) << total_loss / X_train.size() << " | "
                    << "训练集制裁率(Acc): " << fixed << setprecision(2) << (correct * 100.0 / X_train.size()) << " %" << endl;
            }
        }
        
        cout << "\n[!!!] 训练集炼丹大功告成！高维边界已锁定！" << endl;
        
        // 【新增】：训练完了，赶紧把神功保存下来！
        ModelIO::save_model(egg, "egg_net_v1.txt");
    } else {
        cout << "\n[*] 侦测到满级账号，已直接跳过 100 轮训练期！" << endl;
    }
    
    cout << "\n[*] EGG_NET 引擎点火！开始 100 轮地狱梯度下降！" << endl;
    int epochs = 100; 
    
    cout << "\n[!!!] 训练集炼丹大功告成！高维边界已锁定！" << endl;

    cout << "\n[*] 正在准备期末考试卷... 提取后 " << test_size << " 张绝对未知数据的特征..." << endl;
    int test_correct = 0;
    
    for (int i = train_size; i < total_size; i++) {
        MathMagic::apply_fft_filter(dataset[i], 12); 
        MathMagic::apply_tv_denoising(dataset[i], 0.05, 30);
        
        Matrix X_test(787, 1); 
        int idx = 0;
        for(int r = 0; r < 28; r++) {
            for(int c = 0; c < 28; c++) {
                X_test.a[idx++][0] = dataset[i].pixels[r][c] / 255.0;
            }
        }
        X_test.a[idx++][0] = VisionFeatures::count_components(dataset[i]) / 3.0; 
        X_test.a[idx++][0] = VisionFeatures::get_quotient(dataset[i]);           
        X_test.a[idx++][0] = VisionFeatures::curve_enegy(dataset[i]) / 10.0;     
        
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
    cout << "【最终 4000 张大考成绩报告】" << endl;
    cout << "测试样本量: " << test_size << " 张 (网络绝对没见过的数据)" << endl;
    cout << "盲猜成功数: " << test_correct << " 张" << endl;
    cout << "真实泛化准确率: " << fixed << setprecision(2) << (test_correct * 100.0 / test_size) << " %" << endl;
    cout << "======================================" << endl;

    return 0;
}