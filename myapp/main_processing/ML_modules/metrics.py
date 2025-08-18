import os
from scipy import interp
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    """
    计算约登指数并找到最佳阈值
    :param TPR: 真正类率数组
    :param FPR: 假正类率数组
    :param threshold: 阈值数组
    :return: 最佳阈值和对应的约登指数最大值点
    """
    y = TPR - FPR  # 计算约登指数
    J_index = np.argmax(y)  # 找到约登指数最大值的索引
    optimal_threshold = threshold[J_index]  # 最佳阈值
    point = [FPR[J_index], TPR[J_index]]  # 最佳阈值对应的FPR和TPR点
    return optimal_threshold, point

def extract_imp_feat(feature_importances, X_trn):
    # extracted sig. feats
    importance_index = np.argsort(feature_importances)[::-1]
    num_features_to_keep = min(10, len(feature_importances))
    important_indices = importance_index[:num_features_to_keep]
    X_trn_sel = X_trn[:, important_indices]
    return X_trn_sel, important_indices

def cal_youden(y_trn, y_pred_trn):
    fpr, tpr, thresholds = roc_curve(y_trn, y_pred_trn)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

    return optimal_th, optimal_point

def ROC(y_trn, y_pred_trn, params, model, mode, sets):
    fpr, tpr, thresholds = roc_curve(y_trn, y_pred_trn)
    roc_auc = auc(fpr, tpr)
    fpr_interp, tpr_ci_lower, tpr_ci_upper, lower_bound, upper_bound = ROC_bootstrap(y_trn, y_pred_trn)

    # plotting
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='#6a4c93', lw=2,
             label='ROC: %0.3f)' % roc_auc)
    plt.fill_between(fpr_interp, tpr_ci_lower, tpr_ci_upper, color='#3a6ea5', alpha=0.2,
                     label=r'95%% CI: [$%0.3f$, $%0.3f$]' % (lower_bound, upper_bound))
    plt.plot([0, 1], [0, 1], 'k--', color='navy')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18)
    plt.ylabel('True Positive Rate', fontsize=18)
    plt.legend(loc="lower right", fontsize=18)  # 设置图例字体大小
    # plt.show()
    plt.savefig(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML',
                             f'ROC-{model}-ML results-{mode}-{sets}.pdf'), dpi=300,
                bbox_inches='tight')
    return {'AUC': roc_auc, 'AUC_lower': lower_bound, 'AUC_upper': upper_bound}

def ROC_bootstrap(y_trn, y_pred_trn):
    n_bootstraps = 1000
    auc_bootstraps = []
    fpr_boots = []
    tpr_boots = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(y_trn),
                                   size=len(y_trn), replace=True)
        boot_status = y_trn[indices]
        boot_pred = np.array(y_pred_trn)[indices]
        fpr_boot, tpr_boot, _ = roc_curve(boot_status, boot_pred)
        fpr_boots.append(fpr_boot)
        tpr_boots.append(tpr_boot)
        auc_boot = auc(fpr_boot, tpr_boot)
        auc_bootstraps.append(auc_boot)
    # 计算置信区间
    auc_rank = np.array(auc_bootstraps)
    lower_bound = np.percentile(auc_rank, 2.5)
    upper_bound = np.percentile(auc_rank, 97.5)
    # 插值所有bootstrap的ROC曲线到原始FPR的网格上
    fpr_interp = np.linspace(0, 1, 100)
    tpr_interp_all = np.array(
        [interp(fpr_interp, fpr_boot, tpr_boot) for fpr_boot, tpr_boot in
         zip(fpr_boots, tpr_boots)])
    # 计算置信区间的上下界
    tpr_ci_lower = np.percentile(tpr_interp_all, 2.5, axis=0)
    tpr_ci_upper = np.percentile(tpr_interp_all, 97.5, axis=0)

    return fpr_interp, tpr_ci_lower, tpr_ci_upper, lower_bound, upper_bound

def compute_VIP(X, y, R, T, A):
    """
    计算模型中各预测变量的VIP值
    :param X: 数据集X
    :param y: 标签y
    :param R: A个PLS成分中，每个成分a都对应一套系数wa将X转换为成分得分，系数矩阵写作R，大小为p×A
    :param T: 得分矩阵记做T，大小为n×A，ta代表n个样本的第a个成分的得分列表
    :param A: PLS成分的总数
    :return: VIPs = np.zeros(p)
    """
    p = X.shape[1]
    Q2 = np.square(np.dot(y.T, T))

    VIPs = np.zeros(p)
    temp = np.zeros(A)
    for j in range(p):
        for a in range(A):
            temp[a] = Q2[a] * pow(R[j, a] / np.linalg.norm(R[:, a]), 2)
        VIPs[j] = np.sqrt(p * np.sum(temp) / np.sum(Q2))
    return VIPs