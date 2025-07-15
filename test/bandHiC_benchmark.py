import time
import numpy as np
import pytest
from bandhic import band_hic_matrix
import bandhic as bh
import matplotlib.pyplot as plt
import hicstraw

def benchmark_creation():
    """测试矩阵创建性能"""
    sizes = [100, 1000, 5000]
    results = {}
    
    for size in sizes:
        # 创建密集矩阵
        dense_data = np.ones((size, size))
        
        # 测试创建时间
        start_time = time.time()
        mat = band_hic_matrix(dense_data, diag_num=100)
        creation_time = time.time() - start_time
        
        # 测试内存使用
        memory_usage = mat.memory_usage()
        dense_memory = size * size * 8  # 密集矩阵的内存使用（字节）
        
        results[size] = {
            'creation_time': creation_time,
            'memory_usage': memory_usage,
            'memory_ratio': memory_usage / dense_memory
        }
    
    return results

def benchmark_operations():
    """测试基本操作性能"""
    size = 1000
    mat = band_hic_matrix(np.ones((size, size)), diag_num=100)
    results = {}
    
    # 测试索引操作
    start_time = time.time()
    for i in range(1000):
        _ = mat[i % size, i % size]
    results['indexing'] = time.time() - start_time
    
    # 测试设置值
    start_time = time.time()
    for i in range(1000):
        mat[i % size, i % size] = i
    results['setting'] = time.time() - start_time
    
    # 测试统计操作
    start_time = time.time()
    _ = mat.sum()
    results['sum'] = time.time() - start_time
    
    start_time = time.time()
    _ = mat.mean()
    results['mean'] = time.time() - start_time
    
    return results

def benchmark_large_operations():
    """测试大规模操作性能"""
    size = 5000
    mat = band_hic_matrix(np.ones((size, size)), diag_num=500)
    results = {}
    
    # 测试大规模切片
    start_time = time.time()
    _ = mat[1000:2000, 1000:2000]
    results['large_slice'] = time.time() - start_time
    
    # 测试大规模掩码操作
    start_time = time.time()
    mat.add_mask(np.arange(1000), np.arange(1000))
    results['large_mask'] = time.time() - start_time
    
    # 测试大规模转换
    start_time = time.time()
    _ = mat.todense()
    results['to_dense'] = time.time() - start_time
    
    return results

def test_benchmarks():
    """运行所有基准测试"""
    # 创建性能测试
    creation_results = benchmark_creation()
    print("\n创建性能测试结果:")
    for size, result in creation_results.items():
        print(f"大小 {size}x{size}:")
        print(f"  创建时间: {result['creation_time']:.3f}秒")
        print(f"  内存使用: {result['memory_usage']/1024/1024:.2f}MB")
        print(f"  内存比率: {result['memory_ratio']:.2%}")
    
    # 基本操作性能测试
    operation_results = benchmark_operations()
    print("\n基本操作性能测试结果:")
    for op, time_taken in operation_results.items():
        print(f"{op}: {time_taken:.3f}秒")
    
    # 大规模操作性能测试
    large_results = benchmark_large_operations()
    print("\n大规模操作性能测试结果:")
    for op, time_taken in large_results.items():
        print(f"{op}: {time_taken:.3f}秒")
        
# 画一张图，展示不同分辨率下，band_hic_matrix的占用内存和dense矩阵的占用内存的比较
def plot_memory_usage():
    """画一张图，展示不同分辨率下，band_hic_matrix的占用内存和dense矩阵的占用内存的比较"""
    resolutions = [5000, 10000, 25000, 50000, 100000]
    memory_usage = []
    dense_memory = []
    memory_ratio = []
    straw_file = '/Users/wwb/Documents/workspace/BandHiC-Master/data/GSE130275_mESC_WT_combined_1.3B_microc.hic'
    # resolution_list = hicstraw.HiCFile(straw_file).getResolutions()
    # print(resolution_list)
    # resolutions = [resolution<=100000 for resolution in resolution_list]
    for resolution in resolutions:
        print(resolution)
        mat = bh.straw_chr('/Users/wwb/Documents/workspace/BandHiC-Master/data/GSE130275_mESC_WT_combined_1.3B_microc.hic', 'chr1', resolution=resolution, diag_num=2000000//resolution)
        memory_ratio.append(mat.memory_usage()/mat.todense().nbytes)
        memory_usage.append(mat.memory_usage()/1024/1024)
        dense_memory.append(mat.todense().nbytes/1024/1024)
    bar_width = 0.4
    index = np.arange(len(resolutions))
    plt.figure(figsize=(5, 4))
    plt.bar(index, memory_usage, bar_width, label='band_hic_matrix', color='red')
    plt.bar(index + bar_width, dense_memory, bar_width, label='dense_matrix', color='blue')

    plt.xlabel('Resolution')
    plt.ylabel('Memory usage (MB)')
    plt.xticks(index + bar_width / 2, [str(r) for r in resolutions])
    for i, m in enumerate(memory_usage):
        if m>=10:
            plt.text(index[i], m, f'{m:.0f}', ha='center', va='bottom')
        else:
            plt.text(index[i], m, f'{m:.2f}', ha='center', va='bottom')

    for i, m in enumerate(dense_memory):
        if m>=10:
            plt.text(index[i] + bar_width, m, f'{m:.0f}', ha='center', va='bottom')
        else:
            plt.text(index[i] + bar_width, m, f'{m:.2f}', ha='center', va='bottom')
    plt.yscale('log')
    plt.title('Memory usage comparison')
    plt.legend()
    plt.tight_layout()
    plt.savefig('memory_usage_comparison.pdf',format='pdf', dpi=300,bbox_inches='tight')
    plt.show()
    

if __name__ == "__main__":
    test_benchmarks()
    # plot_memory_usage()
