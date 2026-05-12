#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include "data_loader.hpp"

using namespace std;
using db = double;
using pii = pair<int, int>;

namespace VisionFeatures {
    inline constexpr int dx[] = {-1, 1, 0, 0};
    inline constexpr int dy[] = {0, 0, -1, 1};

    inline bool in(int x, int y) {
        return 0 <= x && x < 28 && 0 <= y && y < 28;
    }

    inline db get_fucking_line(const auto& data) {
        int res = -1;
        double best = 0;
        auto v = data.pixels;
        for (int t = 0; t < 255; t ++) {
            vector <int> w (2);
            int sum = 0, grey = 0;
            for (int i = 0; i < 28; i ++) {
                for (int j = 0; j < 28; j ++) {
                    w[v[i][j] > t] ++;
                    sum += v[i][j];
                    grey += (v[i][j] <= t) * v[i][j];
                }
            }
            if (w[0] == 0 || w[1] == 0) continue;
            vector <db> u = {grey * 1.0 / w[0], (sum - grey) * 1.0 / w[1]};
            double g = w[0] * w[1] * (u[0] - u[1]) * (u[0] - u[1]);
            if (res == -1 || (best < g)) {
                res = t;
                best = g;
            }
        }
        return res;
    }

    inline int count_components(const auto& data) {
        auto g = data.pixels;
        int t = get_fucking_line(data);
        vector <vector<bool>> st(28, vector<bool>(28, 0));
        queue <pii> q;
        int res = 0;
        for (int i = 0; i < 28; i ++) {
            for (int j = 0; j < 28; j ++) {
                if (st[i][j] == 0 && g[i][j] <= t) {
                    st[i][j] = 1;
                    q.push({i, j});
                    res ++;
                    while (q.size()) {
                        auto [x, y] = q.front();
                        q.pop();
                        for (int k = 0; k < 4; k ++) {
                            int u = x + dx[k], v = y + dy[k];
                            if (in(u, v) && st[u][v] == 0 && g[u][v] <= t) {
                                st[u][v] = 1;
                                q.push({u, v});
                            }
                        }
                    }
                }
            }
        }
        return res - 1;
    }

    inline db get_quotient(const auto& data) {
        db res = 4.0 * acos(-1.0);
        auto g = data.pixels;
        int s = 0, c = 0, t = get_fucking_line(data);
        for (int i = 0; i < 28; i ++) {
            for (int j = 0; j < 28; j ++) {
                if (g[i][j] > t) {
                    s ++;
                    for (int k = 0; k < 4; k ++) {
                        int u = i + dx[k], v = j + dy[k];
                        if (in(u, v) == 0 || g[u][v] <= t) {
                            c ++;
                            break;
                        }
                    }
                }
            }
        }
        if (c == 0) return 0;
        return res * s * 1.0 / (c * c);
    }

    inline db curve_enegy(const auto& data) {
        auto g = data.pixels;
        int t = get_fucking_line(data), c = 0;
        double res = 0;
        auto lonely_pixel = [&](int i, int j) {
            if (g[i][j] > t) {
                for (int k = 0; k < 4; k ++) {
                    int nx = i + dx[k], ny = j + dy[k];
                    if (in(nx, ny) == 0 || g[nx][ny] <= t) return true;
                }
                return false;
            }
            return false;
        };
        auto get_enegy = [&](int x, int y) {
            db m = 0;
            vector <db> sum(2, 0);
            for (int i = x - 2; i <= x + 2; i ++) {
                for (int j = y - 2; j <= y + 2; j ++) {
                    if (in(i, j) && g[i][j] > t) {
                        m += 1;
                        sum[0] += i;
                        sum[1] += j;
                    }
                }
            }
            if (m == 0) return 0.0;
            return (sum[0] / m - x) * (sum[0] / m - x) + (sum[1] / m - y) * (sum[1] / m - y);
        };
        for (int i = 0; i < 28; i ++) {
            for (int j = 0; j < 28; j ++) {
                if (lonely_pixel(i, j)) {
                    c ++;
                    res += get_enegy(i, j);
                }
            }
        }
        return c ? res / c : 0.0;
    }
}