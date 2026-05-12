#include <bits/stdc++.h>
// 引入图像解析神器！(只需定义一次宏)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include "data_loader.hpp"
#include "math_magic.hpp"
#include "matrix.hpp"
#include "vision_features.hpp"
#include "model_io.hpp"
#include "egg_net.hpp"

using namespace std;

// 1. 解析真实世界的图片为机甲格式
Image load_real_image(const string& path) {
    int w, h, channels;
    // 强制以单通道灰度图 (1) 读取
    unsigned char* data = stbi_load(path.c_str(), &w, &h, &channels, 1);
    
    if (!data || w != 28 || h != 28) {
        cerr << "[!] 致命错误：视网膜对接失败！图片必须存在且为 28x28 像素！" << endl;
        exit(1);
    }

    Image img;
    img.label = -1; // 未知目标
    for (int i = 0; i < 28; i++) {
        for (int j = 0; j < 28; j++) {
            // 将一维的像素流映射回我们的 28x28 矩阵
            img.pixels[i][j] = data[i * 28 + j];
        }
    }
    stbi_image_free(data); // 释放内存
    return img;
}

int main() {
    system("chcp 65001 > nul");
    cout << "=======================================" << endl;
    cout << "   [EGG_NET 实战雷达侦测系统启动]   " << endl;
    cout << "=======================================" << endl;

    // 1. 唤醒机甲
    Egg_Net egg({787, 64, 10});
    if (!ModelIO::load_model(egg, "egg_net_v1.txt")) {
        cout << "[!] 找不到记忆体！请先运行 number_problem 进行训练！" << endl;
        return 0;
    }

    // 2. 载入你手写的图片
    string target_file = "my_number.png";
    cout << "[*] 正在扫描目标文件: " << target_file << " ..." << endl;
    Image my_img = load_real_image(target_file);

    // 3. 必须经过同样的物理净化！（否则机甲不认识真实世界的噪点）
    MathMagic::apply_fft_filter(my_img, 12); 
    MathMagic::apply_tv_denoising(my_img, 0.05, 30);

    // 4. 组装 787 维特征导弹
    Matrix X(787, 1); 
    int idx = 0;
    for(int r = 0; r < 28; r++) {
        for(int c = 0; c < 28; c++) {
            X.a[idx++][0] = my_img.pixels[r][c] / 255.0;
        }
    }
    X.a[idx++][0] = VisionFeatures::count_components(my_img) / 3.0; 
    X.a[idx++][0] = VisionFeatures::get_quotient(my_img);           
    X.a[idx++][0] = VisionFeatures::curve_enegy(my_img) / 10.0;     

    // 5. 主炮开火：前向传播推理！
    Matrix Pred = egg.forward(X);
    
    // 6. 提取最大概率
    int best_guess = 0;
    double max_p = Pred.a[0][0];
    for(int k = 1; k < 10; k++) {
        if(Pred.a[k][0] > max_p) {
            max_p = Pred.a[k][0];
            best_guess = k;
        }
    }

    cout << "\n[!!!] 扫描完毕！高维坍缩确定！" << endl;
    cout << "---------------------------------------" << endl;
    cout << "机甲认为你写的数字是：【 " << best_guess << " 】" << endl;
    cout << "自信度指标 (神经元激活值): " << fixed << setprecision(4) << max_p << endl;
    cout << "---------------------------------------" << endl;

    return 0;
}