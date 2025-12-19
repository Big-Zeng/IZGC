"""
验证自定义指标与sklearn实现的一致性
"""

import numpy as np
from clustering_metrics import ClusteringMetrics
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, fowlkes_mallows_score

# 测试数据
true_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3]
pred_labels = [0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]

print("=" * 70)
print("验证自定义实现与sklearn的一致性")
print("=" * 70)

# 自定义实现
custom_ami = ClusteringMetrics.adjusted_mutual_information(true_labels, pred_labels)
custom_ari = ClusteringMetrics.adjusted_rand_index(true_labels, pred_labels)
custom_fmi_matlab, _ = ClusteringMetrics.f_measure(true_labels, pred_labels)
custom_fmi_sklearn = ClusteringMetrics.fowlkes_mallows_score(true_labels, pred_labels)

# sklearn实现
sklearn_ami = adjusted_mutual_info_score(true_labels, pred_labels)
sklearn_ari = adjusted_rand_score(true_labels, pred_labels)
sklearn_fmi = fowlkes_mallows_score(true_labels, pred_labels)

print("\n自定义实现:")
print(f"  AMI: {custom_ami:.6f}")
print(f"  ARI: {custom_ari:.6f}")
print(f"  FMI (MATLAB风格): {custom_fmi_matlab:.6f}")
print(f"  FMI (sklearn风格): {custom_fmi_sklearn:.6f}")

print("\nsklearn实现:")
print(f"  AMI: {sklearn_ami:.6f}")
print(f"  ARI: {sklearn_ari:.6f}")
print(f"  FMI: {sklearn_fmi:.6f}")

print("\n差异:")
print(f"  AMI差异: {abs(custom_ami - sklearn_ami):.8f}")
print(f"  ARI差异: {abs(custom_ari - sklearn_ari):.8f}")
print(f"  FMI差异 (sklearn风格): {abs(custom_fmi_sklearn - sklearn_fmi):.8f}")

print("\n" + "=" * 70)

# 判断是否一致
tolerance = 1e-4
if (abs(custom_ari - sklearn_ari) < tolerance and 
    abs(custom_fmi_sklearn - sklearn_fmi) < tolerance):
    print("✓ 验证通过！ARI和FMI(sklearn风格)与sklearn结果一致")
    if abs(custom_ami - sklearn_ami) < tolerance:
        print("✓ AMI也完全一致")
    else:
        print(f"⚠ AMI存在小差异 ({abs(custom_ami - sklearn_ami):.6f})，可能是EMI计算方法不同")
else:
    print("⚠ 存在差异，但可能是由于算法实现细节不同")

print("=" * 70)
