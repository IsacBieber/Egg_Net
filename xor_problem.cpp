#include <bits/stdc++.h>
#define eps 1e-7
using namespace std;
using db = double;
using ll = long long;
using ull = unsigned long long;
mt19937_64 rnd(chrono::steady_clock::now().time_since_epoch().count());
uniform_real_distribution<double> dist(-1.0, 1.0);
struct Matrix{
    int row, column;
    vector <vector<db>> a;

    Matrix() {

    }

    Matrix(int row, int column) : row(row), column(column) {
        a.resize(row);
        for (int i = 0; i < row; i ++) {
            a[i].resize(column);
        }
    }

    void set_size(int r, int c) {
        row = r, column = c;
        a.resize(r);
        for (int i = 0; i < r; i ++) {
            a[i].resize(c);
        }
    }

    void randomize() {
        for (int i = 0; i < row; i ++) {
            for (int j = 0; j < column; j ++) {
                a[i][j] = dist(rnd);
            }
        }
    }

    void all_set(db k) {
        for (int i = 0; i < row; i ++) {
            for (int j = 0; j < column; j ++) {
                a[i][j] = k;
            }
        }
    }

    Matrix operator*(Matrix b) {
        Matrix res (row, b.column);
        for (int i = 0; i < row; i ++) {
            for (int j = 0; j < b.column; j ++) {
                for (int k = 0; k < column; k ++) {
                    res.a[i][j] += a[i][k] * b.a[k][j];
                }
            }
        }

        return res;
    }

    Matrix operator*(db v) {
        Matrix res (row, column);
        for (int i = 0; i < row; i ++) {
            for (int j = 0; j <  column; j ++) {
                res.a[i][j] = a[i][j] * v;
            }
        }

        return res;
    }

    Matrix operator+(Matrix b) {
        Matrix res (row, column);
        for (int i = 0; i < row; i ++) {
            for (int j = 0; j < column; j ++) {
                res.a[i][j] =  a[i][j] + b.a[i][j];
            }
        }

        return res;
    }

    Matrix operator-(Matrix b) {
        Matrix res (row, column);
        for (int i = 0; i < row; i ++) {
            for (int j = 0; j < column; j ++) {
                res.a[i][j] =  a[i][j] - b.a[i][j];
            }
        }

        return res;
    }

    Matrix hadamard_product(Matrix b) {
        Matrix res (row, column);
        for (int i = 0; i < row; i ++) {
            for (int j = 0; j < column; j ++) {
                res.a[i][j] =  a[i][j] * b.a[i][j];
            }
        }

        return res;
    }

    Matrix transpose() {
        Matrix res(column, row); 
        for (int i = 0; i < row; i ++) {
            for (int j = 0; j < column; j ++) {
                res.a[j][i] = a[i][j]; 
            }
        }
        
        return res;
    }
};
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
            W.randomize();
        }
    }

    void set_learning_rate(db v) {
        learning_rate = v;
    }

    Matrix process(Matrix X) {
        Matrix res = A = X;
        if (linear) { // A = W * X + B
            res = W * res + B;
        } else { // RELU
            for (int i = 0; i < res.row; i ++) {
                res.a[i][0] = max(res.a[i][0], 0.0);
            }
        }

        return res;
    }

    Matrix adjust(Matrix G) {
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
    Matrix ans;
    Egg_Net () {}
    Egg_Net (int depth) : depth(depth) {
        l.resize(depth);
        for (int i = 0; i < depth; i ++) {
            if (0 < i && i < depth - 1) l[i].linear = !(i & 1);
            else l[i].linear = true; 
            
            int in_dim = 4, out_dim = 4;
            if (i == 0) in_dim = 2;          
            if (i == depth - 1) out_dim = 1; 

            // Layer参数是 (out_dim, in_dim, linear)
            l[i] = Layer(out_dim, in_dim, l[i].linear); 
            l[i].set_learning_rate(0.05); 
        }
    }

    Matrix forward(Matrix X) 
    {
        Matrix ans = X;
        for (int i = 0; i < depth; i ++) {// iterator Layber by Layer
            ans = l[i].process(ans);
        }

        return ans;
    }

    void backward(Matrix G) 
    {
        for (int i = depth - 1; i >= 0; i --) {
            G = l[i].adjust(G);
        }
    }
};
Matrix generate_data() 
{
    Matrix res(2, 1);
    for (int i = 0; i < 2; i ++) {
        res.a[i][0] = rnd() % 2;
    }
    return res;
}
Matrix generate_answer(Matrix in) 
{
    Matrix res(1, 1);
    res.a[0][0] = (int(in.a[0][0]) ^ int(in.a[1][0])); 
    return res;
}
db get_loss(Matrix diff) 
{
    diff = diff.hadamard_product(diff);
    Matrix b = Matrix(1, diff.row);
    b.all_set(0.5);
    return (b * diff).a[0][0];
}
void training(Egg_Net &egg, int up) 
{
    for (int step = 1; step <= up; step ++) {
        Matrix X = generate_data(), Y = generate_answer(X), Pred = egg.forward(X);
        Matrix G = Pred - Y; 
        egg.backward(G); 
        if (step % 1000 == 0) {
            cout << "Step: " << step << " | Loss: " << get_loss(G) << endl;
        }
    }
}
int main(void)
{
    Egg_Net egg (3);
    cout << "Engine Start! Training..." << endl;
    training(egg, 10000); 

    cout << "\n--- Training Done. Testing ---" << endl;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            Matrix X(2, 1); 
            X.a[0][0] = i; 
            X.a[1][0] = j;
            Matrix Pred = egg.forward(X);
            
            double out_val = Pred.a[0][0];
            int real_ans = i ^ j; 

            if (abs(out_val - real_ans) <= eps) {
                cout << i << " XOR " << j << " = " << real_ans << "  [Accept!]" << endl;
            } else {
                cout << i << " XOR " << j << " = " << out_val << "  [Wrong Answer!]" << endl;
            }
        }
    }

    
    return 0;
}