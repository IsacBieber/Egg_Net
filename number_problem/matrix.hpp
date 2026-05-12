#pragma once
#include <vector>

using namespace std;
using db = double;

struct Matrix {
    int row, column;
    vector<vector<db>> a;

    Matrix() {}
    Matrix(int row, int column) : row(row), column(column) {
        a.resize(row, vector<db>(column, 0));
    }

    void set_size(int r, int c) {
        row = r, column = c;
        a.assign(r, vector<db>(c, 0));
    }

    void all_set(db k) {
        for (int i = 0; i < row; i ++)
            for (int j = 0; j < column; j ++)
                a[i][j] = k;
    }

    // 性能优化：const Matrix& 避免深拷贝！
    Matrix operator*(const Matrix& b) const {
        Matrix res(row, b.column);
        for (int i = 0; i < row; i ++)
            for (int j = 0; j < b.column; j ++)
                for (int k = 0; k < column; k ++)
                    res.a[i][j] += a[i][k] * b.a[k][j];
        return res;
    }

    Matrix operator*(db v) const {
        Matrix res(row, column);
        for (int i = 0; i < row; i ++)
            for (int j = 0; j < column; j ++)
                res.a[i][j] = a[i][j] * v;
        return res;
    }

    Matrix operator+(const Matrix& b) const {
        Matrix res(row, column);
        for (int i = 0; i < row; i ++)
            for (int j = 0; j < column; j ++)
                res.a[i][j] = a[i][j] + b.a[i][j];
        return res;
    }

    Matrix operator-(const Matrix& b) const {
        Matrix res(row, column);
        for (int i = 0; i < row; i ++)
            for (int j = 0; j < column; j ++)
                res.a[i][j] = a[i][j] - b.a[i][j];
        return res;
    }

    Matrix transpose() const {
        Matrix res(column, row); 
        for (int i = 0; i < row; i ++)
            for (int j = 0; j < column; j ++)
                res.a[j][i] = a[i][j]; 
        return res;
    }
};