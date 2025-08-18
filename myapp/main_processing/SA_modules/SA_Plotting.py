import os
import numpy as np
import pandas as pd
from scipy import interp
from sklearn.metrics import roc_curve, auc
import seaborn as sns
import matplotlib.pyplot as plt

def SA_Plotting(params, trn_pred,all_Trn_dat, mode, model, sets):
    year_dict = {
        'one_year': 1 * 365,
        'three_year': 3 * 365,
        'five_year': 5 * 365,
        'ten_year': 10 * 365,
    }
    max_index = trn_pred.index.max()
    filtered_year_dict = {k: v for k, v in year_dict.items() if v <= max_index}
    ROC_res = {}
    for k, v in filtered_year_dict.items():
        ## Trn dat
        # extracted dat closed to filter years
        closest_index = trn_pred.index[abs(trn_pred.index.values - v).argmin()]
        Sel_trn_dat = trn_pred.loc[[closest_index]]
        fpr, tpr, _ = roc_curve(all_Trn_dat['status'], Sel_trn_dat.values.flatten())
        # roc
        roc_auc = auc(fpr, tpr)
        # Bootstrap重采样计算AUC
        fpr_interp, tpr_ci_lower, tpr_ci_upper,lower_bound, upper_bound = SA_roc_bootstrap(all_Trn_dat, Sel_trn_dat)

        # 绘制ROC曲线和置信区间
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
        plt.savefig(os.path.join(params['Parent_FilePath'], params['project_name']
, 'Results/SA',
                                 f'{model}-plot-SA-results-ROC-{k}-{mode}-{sets}.pdf'), dpi=300,
                    bbox_inches='tight')
        ROC_res.update({k:{'AUC':roc_auc,'AUC_lower':lower_bound, 'AUC_upper':upper_bound}})
    return ROC_res

def SA_roc_bootstrap(all_dat, Sel_dat):
    n_bootstraps = 1000
    auc_bootstraps = []
    fpr_boots = []
    tpr_boots = []
    for _ in range(n_bootstraps):
        indices = np.random.choice(len(all_dat['status']),
                                   size=len(all_dat['status']), replace=True)
        boot_status = all_dat['status'][indices]
        boot_pred = Sel_dat.values.flatten()[indices]
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

    return fpr_interp, tpr_ci_lower, tpr_ci_upper,lower_bound, upper_bound

def RSF_Plotting(params, y_trn, all_Trn_dat, risk_train,mode, model, sets):
    # roc
    column_names = [f'risk_score_{i + 1}' for i in range(len(risk_train))]
    risk_train_df = pd.DataFrame([risk_train], columns=column_names)

    fpr, tpr, _ = roc_curve(y_trn['status'], risk_train)
    roc_auc = auc(fpr, tpr)
    fpr_interp, tpr_ci_lower, tpr_ci_upper, lower_bound, upper_bound = SA_roc_bootstrap(all_Trn_dat,
                                                                                        risk_train_df)
    # 绘制ROC曲线和置信区间
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
    plt.savefig(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA',
                             f'{model}-plot-SA-results-ROC-{mode}-{sets}.pdf'), dpi=300,
                bbox_inches='tight')
    return {'AUC':roc_auc,'AUC_lower':lower_bound, 'AUC_upper':upper_bound}

def RSF_Plotting(params, y_trn, all_Trn_dat, risk_train,mode, model, sets):
    # roc
    column_names = [f'risk_score_{i + 1}' for i in range(len(risk_train))]
    risk_train_df = pd.DataFrame([risk_train], columns=column_names)

    fpr, tpr, _ = roc_curve(y_trn['status'], risk_train)
    roc_auc = auc(fpr, tpr)
    fpr_interp, tpr_ci_lower, tpr_ci_upper, lower_bound, upper_bound = SA_roc_bootstrap(all_Trn_dat,
                                                                                        risk_train_df)
    # 绘制ROC曲线和置信区间
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
    plt.savefig(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA',
                             f'{model}-plot-SA-results-ROC-{mode}-{sets}.pdf'), dpi=300,
                bbox_inches='tight')
    return {'AUC':roc_auc,'AUC_lower':lower_bound, 'AUC_upper':upper_bound}

