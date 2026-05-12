import os
import urllib.request
import gzip
import struct

def fetch_and_prepare_mnist():
    # 使用 Google 的稳定镜像源
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_img": "train-images-idx3-ubyte.gz",
        "train_lbl": "train-labels-idx1-ubyte.gz",
        "test_img": "t10k-images-idx3-ubyte.gz",
        "test_lbl": "t10k-labels-idx1-ubyte.gz"
    }

    # 1. 爬取下载
    print("[*] 正在下载原始数据集...")
    for name, filename in files.items():
        if not os.path.exists(filename):
            print(f"    -> Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filename)
        else:
            print(f"    -> {filename} 已存在，跳过下载。")

    # 2. 暴力解析二进制并写入 CSV
    def convert_to_csv(img_file, lbl_file, out_csv):
        print(f"\n[*] 正在将 {img_file} 转换并压入 {out_csv}...")
        with gzip.open(img_file, 'rb') as f_img, \
             gzip.open(lbl_file, 'rb') as f_lbl, \
             open(out_csv, 'w') as f_out:

            # 读取魔数和头信息 (使用 > 大端字节序解析)
            magic_lbl, num_lbl = struct.unpack(">II", f_lbl.read(8))
            magic_img, num_img, rows, cols = struct.unpack(">IIII", f_img.read(16))

            assert num_lbl == num_img, "警告：标签数量和图片数量对不上！"
            
            # Python3 里 f_img.read 返回的 bytes 迭代出来直接就是 0-255 的整数，巨爽！
            for i in range(num_img):
                label = ord(f_lbl.read(1))
                pixels = f_img.read(rows * cols) 
                
                # 拼接格式: label, p1, p2 ... p784
                row_str = f"{label}," + ",".join(map(str, pixels)) + "\n"
                f_out.write(row_str)

                if (i + 1) % 10000 == 0:
                    print(f"    已处理 {i + 1}/{num_img} 张图片...")

    # 开工转换
    convert_to_csv(files["train_img"], files["train_lbl"], "mnist_train.csv")
    convert_to_csv(files["test_img"], files["test_lbl"], "mnist_test.csv")
    print("\n[+] 全机甲弹药装填完毕！C++ 随时可以开火！")

if __name__ == "__main__":
    fetch_and_prepare_mnist()