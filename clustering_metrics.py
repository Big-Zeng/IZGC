"""
聚类评估指标计算模块
包含 AMI (Adjusted Mutual Information)、ARI (Adjusted Rand Index)、FMI (F-Measure) 的实现
"""

import numpy as np
from scipy.special import comb
from collections import Counter


class ClusteringMetrics:
    """聚类评估指标类"""
    
    @staticmethod
    def contingency_table(true_labels, pred_labels):
        """
        构建混淆矩阵（列联表）
        
        参数:
            true_labels: 真实标签数组
            pred_labels: 预测标签数组
            
        返回:
            contingency_matrix: 混淆矩阵
            true_classes: 真实类别列表
            pred_classes: 预测类别列表
        """
        true_labels = np.array(true_labels)
        pred_labels = np.array(pred_labels)
        
        true_classes = np.unique(true_labels)
        pred_classes = np.unique(pred_labels)
        
        n_true = len(true_classes)
        n_pred = len(pred_classes)
        
        contingency = np.zeros((n_true, n_pred), dtype=np.int64)
        
        for i, true_class in enumerate(true_classes):
            for j, pred_class in enumerate(pred_classes):
                contingency[i, j] = np.sum((true_labels == true_class) & 
                                           (pred_labels == pred_class))
        
        return contingency, true_classes, pred_classes
    
    @staticmethod
    def adjusted_rand_index(true_labels, pred_labels):
        """
        计算调整兰德指数 (Adjusted Rand Index, ARI)
        
        参数:
            true_labels: 真实标签
            pred_labels: 预测标签
            
        返回:
            ARI值，范围[-1, 1]，1表示完全匹配
        """
        contingency, _, _ = ClusteringMetrics.contingency_table(true_labels, pred_labels)
        
        # 计算各项统计量
        n = np.sum(contingency)
        
        # 行和列的和
        sum_rows = np.sum(contingency, axis=1)
        sum_cols = np.sum(contingency, axis=0)
        
        # 计算组合数
        # C(n_ij, 2) 的和
        sum_comb_c = np.sum(comb(contingency, 2))
        
        # C(a_i, 2) 的和
        sum_comb_rows = np.sum(comb(sum_rows, 2))
        
        # C(b_j, 2) 的和
        sum_comb_cols = np.sum(comb(sum_cols, 2))
        
        # C(n, 2)
        comb_n = comb(n, 2)
        
        # 计算期望值
        expected_index = sum_comb_rows * sum_comb_cols / comb_n
        
        # 计算最大值
        max_index = (sum_comb_rows + sum_comb_cols) / 2.0
        
        # 计算ARI
        if max_index - expected_index == 0:
            return 0.0
        
        ari = (sum_comb_c - expected_index) / (max_index - expected_index)
        
        return ari

    @staticmethod
    def mutual_information(true_labels, pred_labels):
        """
        计算互信息 (Mutual Information, MI)
        
        参数:
            true_labels: 真实标签
            pred_labels: 预测标签
            
        返回:
            MI值
        """
        contingency, _, _ = ClusteringMetrics.contingency_table(true_labels, pred_labels)
        
        n = np.sum(contingency)
        
        # 行和列的和
        sum_rows = np.sum(contingency, axis=1)
        sum_cols = np.sum(contingency, axis=0)
        
        # 计算MI
        mi = 0.0
        for i in range(contingency.shape[0]):
            for j in range(contingency.shape[1]):
                if contingency[i, j] > 0:
                    mi += contingency[i, j] * np.log(
                        (contingency[i, j] * n) / (sum_rows[i] * sum_cols[j])
                    )
        
        mi = mi / n
        
        return mi
    
    @staticmethod
    def entropy(labels):
        """
        计算熵
        
        参数:
            labels: 标签数组
            
        返回:
            熵值
        """
        n = len(labels)
        label_counts = Counter(labels)
        
        entropy_val = 0.0
        for count in label_counts.values():
            if count > 0:
                prob = count / n
                entropy_val -= prob * np.log(prob)
        
        return entropy_val
    
    @staticmethod
    def adjusted_mutual_information(true_labels, pred_labels):
        """
        计算调整互信息 (Adjusted Mutual Information, AMI)
        
        参数:
            true_labels: 真实标签
            pred_labels: 预测标签
            
        返回:
            AMI值，范围[-1, 1]
        """
        contingency, _, _ = ClusteringMetrics.contingency_table(true_labels, pred_labels)
        
        n = np.sum(contingency)
        
        # 行和列的和
        sum_rows = np.sum(contingency, axis=1)
        sum_cols = np.sum(contingency, axis=0)
        
        # 计算MI
        mi = ClusteringMetrics.mutual_information(true_labels, pred_labels)
        
        # 计算熵
        h_true = ClusteringMetrics.entropy(true_labels)
        h_pred = ClusteringMetrics.entropy(pred_labels)
        
        # 计算期望互信息 (EMI)
        emi = 0.0
        for i in range(len(sum_rows)):
            for j in range(len(sum_cols)):
                min_nij = max(1, sum_rows[i] + sum_cols[j] - n)
                max_nij = min(sum_rows[i], sum_cols[j])
                
                for nij in range(int(min_nij), int(max_nij) + 1):
                    # 计算超几何分布的概率
                    prob = (comb(sum_rows[i], nij, exact=True) * 
                           comb(n - sum_rows[i], sum_cols[j] - nij, exact=True) / 
                           comb(n, sum_cols[j], exact=True))
                    
                    if prob > 0 and nij > 0:
                        emi += (nij / n) * np.log((nij * n) / (sum_rows[i] * sum_cols[j])) * prob
        
        # 计算AMI
        denominator = max(h_true, h_pred) - emi
        
        if abs(denominator) < 1e-10:
            # 如果分母接近0，返回NMI
            if h_true * h_pred == 0:
                return 0.0
            return mi / np.sqrt(h_true * h_pred)
        
        ami = (mi - emi) / denominator
        
        return ami

    @staticmethod
    def f_measure(true_labels, pred_labels):
        """
        计算F-Measure (基于MATLAB实现)
        
        参数:
            true_labels: 真实标签
            pred_labels: 预测标签
            
        返回:
            f_measure: F-Measure值
            accuracy: 准确率
        """
        contingency, true_classes, pred_classes = ClusteringMetrics.contingency_table(
            true_labels, pred_labels
        )
        
        n = np.sum(contingency)
        
        # 计算每个真实类别的样本数
        sum_rows = np.sum(contingency, axis=1)  # P_j
        sum_cols = np.sum(contingency, axis=0)  # C_i
        
        # 计算precision和recall
        # precision[i,j] = contingency[i,j] / sum_cols[j]
        # recall[i,j] = contingency[i,j] / sum_rows[i]
        
        precision = np.zeros_like(contingency, dtype=float)
        recall = np.zeros_like(contingency, dtype=float)
        f_matrix = np.zeros_like(contingency, dtype=float)
        
        for i in range(contingency.shape[0]):
            for j in range(contingency.shape[1]):
                if sum_cols[j] > 0:
                    precision[i, j] = contingency[i, j] / sum_cols[j]
                if sum_rows[i] > 0:
                    recall[i, j] = contingency[i, j] / sum_rows[i]
                
                # 计算F值
                if precision[i, j] + recall[i, j] > 0:
                    f_matrix[i, j] = (2 * precision[i, j] * recall[i, j] / 
                                     (precision[i, j] + recall[i, j]))
        
        # 对每个真实类别，选择最大的F值
        max_f_per_class = np.max(f_matrix, axis=1)
        
        # 加权平均
        weights = sum_rows / n
        f_measure = np.sum(weights * max_f_per_class)
        
        # 计算准确率
        max_per_pred_class = np.max(contingency, axis=0)
        accuracy = np.sum(max_per_pred_class) / n
        
        return f_measure, accuracy
    
    @staticmethod
    def fowlkes_mallows_score(true_labels, pred_labels):
        """
        计算Fowlkes-Mallows指数 (与sklearn一致的实现)
        FMI = sqrt(precision * recall) = TP / sqrt((TP+FP) * (TP+FN))
        
        参数:
            true_labels: 真实标签
            pred_labels: 预测标签
            
        返回:
            FMI值
        """
        contingency, _, _ = ClusteringMetrics.contingency_table(true_labels, pred_labels)
        
        # 计算TP (True Positives): 同一真实类别且同一预测类别的点对数
        tp = np.sum(comb(contingency, 2))
        
        # 计算FP (False Positives): 不同真实类别但同一预测类别的点对数
        # FP = sum_j C(n_j, 2) - TP
        sum_cols = np.sum(contingency, axis=0)
        fp = np.sum(comb(sum_cols, 2)) - tp
        
        # 计算FN (False Negatives): 同一真实类别但不同预测类别的点对数
        # FN = sum_i C(n_i, 2) - TP
        sum_rows = np.sum(contingency, axis=1)
        fn = np.sum(comb(sum_rows, 2)) - tp
        
        # 计算FMI
        denominator = np.sqrt((tp + fp) * (tp + fn))
        
        if denominator == 0:
            return 0.0
        
        fmi = tp / denominator
        
        return fmi
    
    @staticmethod
    def evaluate_all(true_labels, pred_labels, use_sklearn_fmi=True):
        """
        计算所有评估指标
        
        参数:
            true_labels: 真实标签
            pred_labels: 预测标签
            use_sklearn_fmi: 是否使用sklearn风格的FMI (Fowlkes-Mallows)
            
        返回:
            dict: 包含所有指标的字典
        """
        ami = ClusteringMetrics.adjusted_mutual_information(true_labels, pred_labels)
        ari = ClusteringMetrics.adjusted_rand_index(true_labels, pred_labels)
        
        if use_sklearn_fmi:
            fmi = ClusteringMetrics.fowlkes_mallows_score(true_labels, pred_labels)
            accuracy = None
        else:
            fmi, accuracy = ClusteringMetrics.f_measure(true_labels, pred_labels)
        
        result = {
            'AMI': ami,
            'ARI': ari,
            'FMI': fmi,
        }
        
        if accuracy is not None:
            result['Accuracy'] = accuracy
        
        return result


# 示例使用
if __name__ == "__main__":
    # 测试数据
    true_labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
    pred_labels = [0, 0, 1, 1, 1, 1, 2, 2, 2]
    
    print("聚类评估指标测试:")
    print("-" * 50)
    
    # 计算各项指标
    metrics = ClusteringMetrics.evaluate_all(true_labels, pred_labels)
    
    print(f"AMI (Adjusted Mutual Information): {metrics['AMI']:.4f}")
    print(f"ARI (Adjusted Rand Index):         {metrics['ARI']:.4f}")
    print(f"FMI (F-Measure):                   {metrics['FMI']:.4f}")
    print(f"Accuracy:                          {metrics['Accuracy']:.4f}")
    
    print("\n" + "-" * 50)
    print("也可以单独调用各个指标:")
    print(f"AMI: {ClusteringMetrics.adjusted_mutual_information(true_labels, pred_labels):.4f}")
    print(f"ARI: {ClusteringMetrics.adjusted_rand_index(true_labels, pred_labels):.4f}")
    fmi, acc = ClusteringMetrics.f_measure(true_labels, pred_labels)
    print(f"FMI (MATLAB风格): {fmi:.4f}, Accuracy: {acc:.4f}")
    fmi_sklearn = ClusteringMetrics.fowlkes_mallows_score(true_labels, pred_labels)
    print(f"FMI (sklearn风格): {fmi_sklearn:.4f}")
