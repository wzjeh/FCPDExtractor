#!/usr/bin/env python3
"""
手动清理中间文件脚本
用于清理指定目录中的所有中间文件，只保留最终结果
"""
import os
import sys

def cleanup_directory(output_dir):
    """清理指定目录中的中间文件"""
    print("=" * 70)
    print("🗑️  手动清理中间文件")
    print("=" * 70)
    print(f"目标目录: {output_dir}")
    print()
    
    if not os.path.exists(output_dir):
        print(f"❌ 错误: 目录不存在")
        return
    
    # 获取所有文件
    all_files = [f for f in os.listdir(output_dir) if f.endswith('.txt')]
    
    # 查找最终结果文件
    final_files = [f for f in all_files if '_Summarized.txt' in f or '_Overall.txt' in f]
    
    if not final_files:
        print("⚠️  警告: 未找到最终结果文件 (*_Summarized.txt 或 *_Overall.txt)")
        print("   请确保已完成步骤5的处理")
        return
    
    print(f"📦 找到 {len(final_files)} 个最终结果文件:")
    for f in final_files:
        size = os.path.getsize(os.path.join(output_dir, f)) / 1024
        print(f"  • {f} ({size:.1f} KB)")
    
    # 识别中间文件（不是最终结果的文件）
    intermediate_files = [f for f in all_files if f not in final_files]
    
    if not intermediate_files:
        print("\n✅ 目录已经是干净的，没有中间文件")
        return
    
    print(f"\n🗑️  找到 {len(intermediate_files)} 个中间文件:")
    total_size = 0
    for f in intermediate_files:
        size = os.path.getsize(os.path.join(output_dir, f)) / 1024
        total_size += size
        print(f"  • {f} ({size:.1f} KB)")
    
    print(f"\n  总大小: {total_size:.1f} KB")
    
    # 确认删除
    print("\n⚠️  确认删除这些文件？")
    print("  输入 'yes' 继续，或按 Enter 取消")
    confirm = input("  > ").strip().lower()
    
    if confirm != 'yes':
        print("\n❌ 已取消删除")
        return
    
    # 执行删除
    print("\n🗑️  正在删除中间文件...")
    deleted_count = 0
    for f in intermediate_files:
        file_path = os.path.join(output_dir, f)
        try:
            os.remove(file_path)
            print(f"  ✓ 已删除: {f}")
            deleted_count += 1
        except Exception as e:
            print(f"  ✗ 删除失败 {f}: {e}")
    
    print(f"\n🎉 清理完成！删除了 {deleted_count} 个文件，节省 {total_size:.1f} KB")
    print(f"\n📦 最终保留 {len(final_files)} 个文件")

def main():
    if len(sys.argv) > 1:
        # 从命令行参数获取目录
        output_dir = sys.argv[1]
    else:
        # 使用默认目录
        output_dir = '/Users/zhaowenyuan/Projects/FCPDExtractor/Data/output/101021acsoprd7b00291'
    
    cleanup_directory(output_dir)

if __name__ == '__main__':
    main()

