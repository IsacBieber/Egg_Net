#include "data_loader.hpp"
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

// 真正实现这个读取函数
vector<Image> load_mnist_csv(const string& filename, int limit) {
    ifstream file(filename);
    vector<Image> dataset;
    string line;

    if (!file.is_open()) {
        cerr << "Error: 找不到文件 " << filename << endl;
        return dataset;
    }

    while (getline(file, line) && limit != 0) {
        stringstream ss(line);
        string val;
        Image img;

        getline(ss, val, ',');
        img.label = stoi(val);

        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                getline(ss, val, ',');
                img.pixels[i][j] = stoi(val);
            }
        }
        dataset.push_back(img);
        if (limit > 0) limit --;
    }
    return dataset;
}