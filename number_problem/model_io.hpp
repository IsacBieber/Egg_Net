#pragma once
#include <fstream>
#include <iostream>
#include <string>
#include <iomanip>
#include "egg_net.hpp"

namespace ModelIO {
    using namespace std;

    // 1. 将机甲记忆烧录进硬盘
    inline void save_model(const Egg_Net& net, const string& filename) {
        ofstream out(filename);
        if (!out.is_open()) {
            cerr << "[!] 警告：磁盘锁定，无法创建记忆文件！" << endl;
            return;
        }
        
        // 记录层数，作为校验码
        out << net.depth << "\n";
        
        // 遍历所有层，榨干所有的 W 和 B
        for (int i = 0; i < net.depth; i++) {
            const auto& layer = net.l[i];
            
            // 保存 W 矩阵的维度和数据 (保留 8 位精度防止失忆)
            out << layer.W.row << " " << layer.W.column << "\n";
            for (int r = 0; r < layer.W.row; r++) {
                for (int c = 0; c < layer.W.column; c++) {
                    out << fixed << setprecision(8) << layer.W.a[r][c] << " ";
                }
                out << "\n";
            }
            
            // 保存 B 矩阵的维度和数据
            out << layer.B.row << " " << layer.B.column << "\n";
            for (int r = 0; r < layer.B.row; r++) {
                for (int c = 0; c < layer.B.column; c++) {
                    out << fixed << setprecision(8) << layer.B.a[r][c] << " ";
                }
                out << "\n";
            }
        }
        out.close();
        cout << "[+] 机甲突触记忆已成功永久烧录至: " << filename << endl;
    }

    // 2. 从硬盘复活机甲
    inline bool load_model(Egg_Net& net, const string& filename) {
        ifstream in(filename);
        if (!in.is_open()) {
            cout << "[-] 未侦测到记忆卡 (" << filename << ")，将以出厂(白痴)状态启动！" << endl;
            return false;
        }
        
        int expected_depth;
        in >> expected_depth;
        if (expected_depth != net.depth) {
            cerr << "[!] 记忆体版本不匹配！网络结构可能已改变！" << endl;
            return false;
        }
        
        // 逆向吸取数据
        for (int i = 0; i < net.depth; i++) {
            auto& layer = net.l[i];
            int r, c;
            
            // 恢复 W 矩阵
            in >> r >> c;
            for (int x = 0; x < r; x++) {
                for (int y = 0; y < c; y++) {
                    in >> layer.W.a[x][y];
                }
            }
            
            // 恢复 B 矩阵
            in >> r >> c;
            for (int x = 0; x < r; x++) {
                for (int y = 0; y < c; y++) {
                    in >> layer.B.a[x][y];
                }
            }
        }
        in.close();
        cout << "[+] 成功读取记忆体！高维几何特征已加载，机甲满血复活！" << endl;
        return true;
    }
}