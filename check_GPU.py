# check_gpu_detail.py
import torch

print("=" * 50)
print("GPU详细信息")
print("=" * 50)

# 查看GPU数量
print(f"可用GPU数量: {torch.cuda.device_count()}")

for i in range(torch.cuda.device_count()):
    print(f"\nGPU {i}:")
    print(f"  名称: {torch.cuda.get_device_name(i)}")
    print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB")
    print(f"  当前分配显存: {torch.cuda.memory_allocated(i) / 1e9:.2f} GB")
    print(f"  当前保留显存: {torch.cuda.memory_reserved(i) / 1e9:.2f} GB")
    
    # 检查是否可用
    try:
        test_tensor = torch.tensor([1.0]).cuda(i)
        print(f"  状态: ✅ 可用")
    except Exception as e:
        print(f"  状态: ❌ 不可用 - {e}")

print(f"\n当前默认GPU: cuda:{torch.cuda.current_device()}")
print(f"PyTorch CUDA可用: {torch.cuda.is_available()}")
print(f"PyTorch CUDA版本: {torch.version.cuda}")

# 测试两个GPU的性能
if torch.cuda.device_count() >= 2:
    print("\n" + "=" * 50)
    print("多GPU测试")
    print("=" * 50)
    
    # 分别在两个GPU上创建张量
    for i in range(2):
        with torch.cuda.device(i):
            try:
                a = torch.randn(1000, 1000).cuda()
                b = torch.randn(1000, 1000).cuda()
                c = torch.matmul(a, b)
                print(f"GPU {i}: 矩阵乘法测试 ✅ 通过")
            except Exception as e:
                print(f"GPU {i}: 矩阵乘法测试 ❌ 失败 - {e}")