import torch
import time

def benchmark(size=5000):
    print(f"\n>>> Benchmarking matrix multiplication of size {size}x{size}\n")

    # CPU 计算
    x_cpu = torch.rand(size, size, device="cpu")
    y_cpu = torch.rand(size, size, device="cpu")

    torch.cuda.synchronize()
    start = time.time()
    result_cpu = torch.mm(x_cpu, y_cpu)
    end = time.time()
    print(f"CPU time: {end - start:.3f} seconds")

    # GPU 计算
    if torch.cuda.is_available():
        x_gpu = torch.rand(size, size, device="cuda")
        y_gpu = torch.rand(size, size, device="cuda")

        torch.cuda.synchronize()  # 确保前面的操作完成
        start = time.time()
        result_gpu = torch.mm(x_gpu, y_gpu)
        torch.cuda.synchronize()  # 等待 GPU 完成
        end = time.time()
        print(f"GPU time: {end - start:.3f} seconds")
        print(f"GPU device: {result_gpu.device}")
    else:
        print("No CUDA-capable GPU detected.")


if __name__ == "__main__":
    benchmark(5000)
