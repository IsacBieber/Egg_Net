#pragma once // 极客必备：防止头文件被重复包含
#include <vector>
#include <string>

// 数据结构定义
struct Image {
    int label;
    int pixels[28][28];
};

// 声明读取函数，limit 设为 -1 表示默认全读
std::vector<Image> load_mnist_csv(const std::string& filename, int limit = -1);