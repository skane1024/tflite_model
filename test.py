import numpy as np

# 设定参数范围和概率
N_values = np.arange(1, 16)
N_probs = np.where(N_values <= 3, 0.1, 0.01)  # 常见范围的概率更高
N_probs /= N_probs.sum()  # 确保概率之和为1

C_values = np.arange(1, 1024)
C_probs = np.where(C_values <= 512, 0.1, 0.01)
C_probs /= C_probs.sum()

H_W_values = np.arange(1, 2048)
H_W_probs = np.where(H_W_values <= 640, 0.1, 0.01)
H_W_probs /= H_W_probs.sum()

kernel_stride_padding_values = np.arange(1, 5)
kernel_stride_padding_probs = np.full_like(kernel_stride_padding_values, 1.0 / len(kernel_stride_padding_values))  # 均匀分布

# 设定测试用例数量
num_cases = 1000

# 生成随机测试用例
test_cases = []
for _ in range(num_cases):
    N = np.random.choice(N_values, p=N_probs)
    C = np.random.choice(C_values, p=C_probs)
    H = np.random.choice(H_W_values, p=H_W_probs)
    W = np.random.choice(H_W_values, p=H_W_probs)
    
    # 使kernel_h和kernel_w相等的概率更高
    if np.random.rand() < 0.8:  # 80%的概率使它们相等
        kernel_h = kernel_w = np.random.choice(kernel_stride_padding_values, p=kernel_stride_padding_probs)
    else:
        kernel_h = np.random.choice(kernel_stride_padding_values