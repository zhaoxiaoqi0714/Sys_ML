import os
import itertools
import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import roc_curve
from sklearn.ensemble import StackingClassifier
import matplotlib.pyplot as plt
import seaborn as sns

from myapp.tools.ML_RES import ml_matrics
from myapp.pipelines import data_preocessing_in_Uni,load_ML_pre_dat
from myapp.tools.utils import tidy_missing_mean, tidy_missing_MI, tidy_missing_KNN,tidying_missing_test,scaling_dat,proc_Imb

def Ensemble_analysis(params, Single_ML_res, trn_dat, trn_label, test_dat, test_label):
    ## tidying resukts
    opt_missing,opt_scaling,opt_imbalance,best_models_list,pre_mode = prepared_ensemble_dat(Single_ML_res)
    ## pipeline processing
    if len(best_models_list) <= 0:
        print('Only 1 or No ML models showed better performance, therefore do not process ensemble learning.')
    else:
        # check test
        has_missing_values = False
        if test_dat is not None:
            if test_dat.isna().any().any(): has_missing_values = True
        if trn_dat.isna().any().any(): has_missing_values = True
        # tidying mass
        if has_missing_values:
            if opt_missing == 'Mean':
                trn_dat = tidy_missing_mean(trn_dat)
                ensemble_after_missing(params, trn_dat, trn_label, test_dat, test_label, opt_missing, opt_scaling,
                                       opt_imbalance, best_models_list, pre_mode)
            elif opt_missing == 'MI':
                trn_dat = tidy_missing_MI(trn_dat)
                ensemble_after_missing(params, trn_dat, trn_label, test_dat, test_label, opt_missing, opt_scaling,
                                       opt_imbalance, best_models_list, pre_mode)
            elif opt_missing == 'KNN':
                trn_dat = tidy_missing_KNN(trn_dat)
                ensemble_after_missing(params, trn_dat, trn_label, test_dat, test_label, opt_missing, opt_scaling,
                                       opt_imbalance, best_models_list, pre_mode)
        else:
            ensemble_after_missing(params, trn_dat, trn_label, test_dat, test_label, opt_missing, opt_scaling,
                                   opt_imbalance, best_models_list, pre_mode)

def Ensemble_res_ana(params):
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble/Plotting')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble/Plotting'))
    Ensemble_Path = os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble')
    Ensemble_Fileslist = os.listdir(Ensemble_Path)
    Ensemble_Fileslist = [file for file in Ensemble_Fileslist if file.endswith('.csv')]
    Ensemble_RES = ensemble_res_combined(params,Ensemble_Fileslist, Ensemble_Path)
    # save results
    trn_ensemble_res, tst_ensemble_res = ensemble_save_res(params,Ensemble_RES)
    plotting_ensemble(params, trn_ensemble_res, tst_ensemble_res)

def ensemble_after_missing(params, trn_dat, trn_label, test_dat, test_label, opt_missing,opt_scaling,opt_imbalance,best_models_list,pre_mode):
    # ensmeble analysis
    if params['HadLabel']:
        ## prepated training dat
        trn_dat, test_dat, X_tst = load_ML_pre_dat(params, opt_missing, opt_scaling)
        trn_dat, test_dat, X_trn, X_tst, y_trn, y_tst, le = data_preocessing_in_Uni(params, trn_dat, test_dat,
                                                                                    X_tst, trn_label, test_label,
                                                                                    opt_missing, opt_scaling)
        if params['Imbalance']: X_trn, y_trn = proc_Imb(opt_imbalance, X_trn, y_trn)
        if params['HadTest']:
            ## prepared initial model params
            models_param_grid = gen_models_param_grid(X_trn, y_trn)
            best_params = model_best_params(models_param_grid, X_trn, y_trn)
            ## obtained comparison
            all_combinations = gen_all_combinations(best_models_list)
            ## processing ensemble
            ensemble_test(params, X_trn, y_trn, X_tst, test_label, le, all_combinations, models_param_grid, best_params,
                          pre_mode)
        else:
            ## prepared initial model params
            models_param_grid = gen_models_param_grid(X_trn, y_trn)
            best_params = model_best_params(models_param_grid, X_trn, y_trn)
            ## obtained comparison
            all_combinations = gen_all_combinations(best_models_list)
            ## processing ensemble
            ensemble_no_test(params, trn_dat, X_trn, y_trn, trn_label, X_tst, test_label, le, all_combinations,
                             models_param_grid,
                             best_params, pre_mode)
    else:
        print('Ensembl anlysis must input label files.')

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

def cal_youden(y_trn, y_pred_trn):
    fpr, tpr, thresholds = roc_curve(y_trn, y_pred_trn)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)

    return optimal_th, optimal_point

def prepared_ensemble_dat(Single_ML_res):
    ##tidying resukts
    split_result = Single_ML_res.index.str.split('_', expand=True).to_frame(index=False)
    split_result.columns = ['Model', 'MissingMethod', 'ScalingMethod', 'ImbalanceMethod']
    split_result.index = Single_ML_res.index
    Single_ML_res[['Model', 'MissingMethod', 'ScalingMethod', 'ImbalanceMethod']] = split_result

    # ranking indicators
    indicators_columns = ['AUC', 'F1_score', 'PR-AUC', 'Accuracy']
    for col in indicators_columns:
        Single_ML_res[f'{col}_Rank'] = Single_ML_res[col].rank(method='min', ascending=False)
    rank_columns = [col for col in Single_ML_res.columns if col.endswith('_Rank')]
    Single_ML_res['Comprehensive_Ranking'] = Single_ML_res[rank_columns].mean(axis=1)

    # extracted opt comparison in each model
    min_ranking_indices = Single_ML_res.groupby('Model')['Comprehensive_Ranking'].idxmin()
    best_models = Single_ML_res.loc[min_ranking_indices]
    best_models = best_models[best_models['Accuracy'] > 0.75]
    best_models_list = []
    opt_missing = opt_scaling = opt_imbalance = pre_mode = None
    if best_models.shape[0] != 0:
        best_models_list = list(best_models['Model'])
        min_ranking_row = best_models.loc[best_models['Comprehensive_Ranking'].idxmin()]
        opt_missing = min_ranking_row['MissingMethod']
        opt_scaling = min_ranking_row['ScalingMethod']
        opt_imbalance = min_ranking_row['ImbalanceMethod']
        pre_mode = str(opt_missing) + '_' + str(opt_scaling) + '_' + str(opt_imbalance)

    return opt_missing,opt_scaling,opt_imbalance,best_models_list,pre_mode

def gen_models_param_grid(X_trn, y_trn):
    # return {
    #                         'RF': {
    #                             'model': RandomForestClassifier(random_state=777),
    #                             'param_grid': {
    #                                 'n_estimators': [100, 200, 300],
    #                                 'max_depth': [None, 10, 20, 30],
    #                                 'min_samples_split': [2, 5, 10],
    #                                 'min_samples_leaf': [1, 2, 4],
    #                                 'max_features': ['auto', 'sqrt', 'log2']
    #                             }
    #                         },
    #                         'LASSO': {
    #                             'model': Lasso(random_state=777),
    #                             'param_grid': {
    #                                 'alpha': np.logspace(-4, 1, 20)  # alpha的范围从10^-4到10^1，共20个点
    #                             }
    #                         },
    #                         'PLSDA': {
    #                             'model': PLSRegression(),
    #                             'param_grid': {
    #                                 'n_components': range(1, min(X_trn.shape[1], 10) + 1)
    #                                 # 动态设置 n_components
    #                             }
    #                         },
    #                         'Xgboost': {
    #                             'model': xgb.XGBRegressor(random_state=777),
    #                             'param_grid': {
    #                                 'n_estimators': [100, 200, 300],
    #                                 'max_depth': [3, 4, 5],
    #                                 'learning_rate': [0.01, 0.1, 0.2],
    #                                 'subsample': [0.8, 1.0],
    #                                 'colsample_bytree': [0.8, 1.0]
    #                             }
    #                         },
    #                         'Logit': {
    #                             'model': LogisticRegression(solver='liblinear'),  # 使用liblinear求解器以支持小数据集
    #                             'param_grid': {
    #                                 'C': [0.01, 0.1, 1, 10, 100],  # 正则化强度的倒数
    #                                 'penalty': ['l1', 'l2']  # 用于正则化的范数
    #                             }
    #                         },
    #                         'GBDT': {
    #                             'model': GradientBoostingClassifier(random_state=42),
    #                             'param_grid': {
    #                                 'n_estimators': [100, 200],  # 树的数量
    #                                 'learning_rate': [0.01, 0.1],  # 学习率
    #                                 'max_depth': [3, 5],  # 树的最大深度
    #                                 'min_samples_split': [2, 5],  # 分割内部节点所需的最小样本数
    #                                 'min_samples_leaf': [1, 2]  # 叶子节点所需的最小样本数
    #                             }
    #                         },
    #                         'LightGBM': {
    #                             'model': lgb.LGBMClassifier(**{
    #                                 'boosting_type': 'gbdt',
    #                                 'objective': 'binary' if len(set(y_trn)) == 2 else 'multiclass',
    #                                 # 根据任务选择
    #                                 'metric': 'mse',  # 或者根据任务选择 'auc', 'mse' 等
    #                                 'learning_rate': 0.1,
    #                                 'num_leaves': 31,
    #                                 'max_depth': -1,
    #                                 'min_data_in_leaf': 20,
    #                                 'min_sum_hessian_in_leaf': 1e-3,
    #                                 'feature_fraction': 0.9,
    #                                 'bagging_fraction': 0.8,
    #                                 'bagging_freq': 5,
    #                                 'verbose': -1
    #                             }),
    #                             'param_grid': {
    #                                 'num_leaves': [20, 31, 50],
    #                                 'learning_rate': [0.01, 0.1, 0.2],
    #                                 'max_depth': [6, 8, 10],
    #                                 'min_data_in_leaf': [10, 20, 30],
    #                                 'min_sum_hessian_in_leaf': [1e-3, 1e-2, 1e-1]
    #                             }
    #                         },
    #                         'Boruta': {
    #                             'model': RandomForestClassifier(n_jobs=-1, class_weight='balanced'),
    #                             'param_grid': {
    #                                 'n_estimators': [100, 200, 300],  # 决策树的数量
    #                                 'max_depth': [None, 5, 10, 20],  # 树的最大深度
    #                                 'min_samples_split': [2, 5, 10],  # 分割内部节点所需的最小样本数
    #                                 'min_samples_leaf': [1, 2, 4],  # 叶子节点所需的最小样本数
    #                                 'max_features': ['auto', 'sqrt', 'log2']  # 寻找最佳分割时要考虑的特征数量
    #                             }
    #                         },
    #                         'GaussianNB': {
    #                             'model': GaussianNB(),
    #                             'param_grid': {
    #                                 'priors': [None, [0.25, 0.75], [0.75, 0.25]]
    #                             }
    #                         },
    #                         'NN': {
    #                             'model': MLPClassifier(max_iter=300),
    #                             'param_grid': {
    #                                 'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # 隐藏层的大小
    #                                 'activation': ['tanh', 'relu'],  # 激活函数
    #                                 'solver': ['sgd', 'adam'],  # 优化算法
    #                                 'alpha': [0.0001, 0.05],  # L2惩罚参数
    #                                 'learning_rate': ['constant', 'adaptive'],  # 学习率
    #                             }
    #                         }
    #                     }
    return {
        'RF': {
            'model': RandomForestClassifier(random_state=777),
            'param_grid': {
                'n_estimators': [100, 200],  # 减少候选值
                'max_depth': [None, 10, 20],  # 减少候选值
                'min_samples_split': [2, 5],  # 减少候选值
                'min_samples_leaf': [1, 2],  # 减少候选值
                'max_features': ['auto', 'sqrt']  # 减少候选值
            }
        },
        'LASSO': {
            'model': Lasso(random_state=777),
            'param_grid': {
                'alpha': np.logspace(-4, 1, 10)  # 减少候选值
            }
        },
        'PLSDA': {
            'model': PLSRegression(),
            'param_grid': {
                'n_components': range(1, min(X_trn.shape[1], 5) + 1)  # 动态调整
            }
        },
        'Xgboost': {
            'model': xgb.XGBRegressor(random_state=777),
            'param_grid': {
                'n_estimators': [100, 200],  # 减少候选值
                'max_depth': [3, 4],  # 减少候选值
                'learning_rate': [0.01, 0.1],  # 减少候选值
                'subsample': [0.8, 1.0],  # 减少候选值
                'colsample_bytree': [0.8, 1.0]  # 减少候选值
            }
        },
        'Logit': {
            'model': LogisticRegression(solver='liblinear'),
            'param_grid': {
                'C': [0.01, 0.1, 1],  # 减少候选值
                'penalty': ['l1', 'l2']  # 保持不变
            }
        },
        'GBDT': {
            'model': GradientBoostingClassifier(random_state=42),
            'param_grid': {
                'n_estimators': [100],  # 减少候选值
                'learning_rate': [0.01, 0.1],  # 减少候选值
                'max_depth': [3],  # 减少候选值
                'min_samples_split': [2],  # 减少候选值
                'min_samples_leaf': [1]  # 减少候选值
            }
        },
        'LightGBM': {
            'model': lgb.LGBMClassifier(**{
                'boosting_type': 'gbdt',
                'objective': 'binary' if len(set(y_trn)) == 2 else 'multiclass',
                'metric': 'mse',
                'learning_rate': 0.1,
                'num_leaves': 31,
                'max_depth': -1,
                'min_data_in_leaf': 20,
                'min_sum_hessian_in_leaf': 1e-3,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1
            }),
            'param_grid': {
                'num_leaves': [20, 31],  # 减少候选值
                'learning_rate': [0.01, 0.1],  # 减少候选值
                'max_depth': [6, 8],  # 减少候选值
                'min_data_in_leaf': [10, 20],  # 减少候选值
                'min_sum_hessian_in_leaf': [1e-3, 1e-2]  # 减少候选值
            }
        },
        'Boruta': {
            'model': RandomForestClassifier(n_jobs=-1, class_weight='balanced'),
            'param_grid': {
                'n_estimators': [100, 200],  # 减少候选值
                'max_depth': [None, 10],  # 减少候选值
                'min_samples_split': [2, 5],  # 减少候选值
                'min_samples_leaf': [1, 2],  # 减少候选值
                'max_features': ['auto', 'sqrt']  # 减少候选值
            }
        },
        'GaussianNB': {
            'model': GaussianNB(),
            'param_grid': {
                'priors': [None, [0.25, 0.75]]  # 减少候选值
            }
        },
        'NN': {
            'model': MLPClassifier(max_iter=300),
            'param_grid': {
                'hidden_layer_sizes': [(50,), (100,)],  # 减少候选值
                'activation': ['relu'],  # 减少候选值
                'solver': ['adam'],  # 减少候选值
                'alpha': [0.0001, 0.05],  # 减少候选值
                'learning_rate': ['constant']  # 减少候选值
            }
        }
    }

def model_best_params(models_param_grid,X_trn, y_trn):
    best_params = {}
    for model_name, model_info in models_param_grid.items():
        model = model_info['model']
        param_grid = model_info['param_grid']
        grid_search = RandomizedSearchCV(estimator=model, param_distributions=param_grid, n_iter=50, cv=5, random_state=777)
        grid_search.fit(X_trn, y_trn)
        best_param = grid_search.best_params_
        best_params[model_name] = best_param

    return best_params

def gen_all_combinations(best_models_list):
    all_combinations = []
    for r in range(2, len(best_models_list) + 1):
        for combo in itertools.combinations(best_models_list, r):
            all_combinations.append(combo)

    return all_combinations

def ensemble_test(params, X_trn, y_trn, X_tst, test_label, le, all_combinations, models_param_grid, best_params, pre_mode):
    for com in all_combinations:
        model_ensem = '+'.join(com)
        base_classifiers = []
        for bm in com:
            model_instance = models_param_grid[bm]['model'].set_params(
                **best_params[bm])
            base_classifiers.append((bm, model_instance))
            # 定义元分类器
        meta_classifier = LogisticRegression()
        stacking_regressor = StackingClassifier(estimators=base_classifiers,
                                                final_estimator=meta_classifier, cv=5)
        stacking_regressor.fit(X_trn, y_trn)

        # predicted
        # prob
        y_trn_prob = stacking_regressor.predict_proba(X_trn)[:, 1]
        y_test_prob = stacking_regressor.predict_proba(X_tst)[:, 1]
        optimal_th, optimal_point = cal_youden(y_trn, y_trn_prob)
        # pred
        y_pred_trn = [1 if val >= optimal_th else 0 for val in y_trn_prob]
        y_pred_tst = [1 if val >= optimal_th else 0 for val in y_test_prob]

        if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble')):
            os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble'))
        # predicted_values
        trn_res = pd.DataFrame({
            'True_label': le.inverse_transform(y_trn),
            'Pred_label': le.inverse_transform(y_pred_trn),
            'Pred_Prob': y_trn_prob
        })
        tst_res = pd.DataFrame({
            'True_label': test_label['group'],
            'Pred_label': le.inverse_transform(y_pred_tst),
            'Pred_Prob': y_test_prob
        })
        trn_res.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble',
                                    f'Trn dat-Ensemble results-{model_ensem}-{pre_mode}.csv'),
                       index=None)
        tst_res.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble',
                                    f'Test dat-Ensemble results-{model_ensem}-{pre_mode}.csv'),
                       index=None)

def ensemble_no_test(params,trn_dat, X_trn, y_trn,trn_label, X_tst, test_label, le, all_combinations, models_param_grid, best_params, pre_mode):
    for com in all_combinations:
        model_ensem = '+'.join(com)
        base_classifiers = []
        for bm in com:
            model_instance = models_param_grid[bm]['model'].set_params(
                **best_params[bm])
            base_classifiers.append((bm, model_instance))
            # 定义元分类器
        meta_classifier = LogisticRegression()
        stacking_regressor = StackingClassifier(estimators=base_classifiers,
                                                final_estimator=meta_classifier, cv=5)
        # obtained results of each cv
        kf = KFold(n_splits=params['CV'], shuffle=True, random_state=777)
        y_pred_trn_folds, y_prob_trn_folds, y_true_trn_folds, y_pred_tst_folds, y_prob_tst_folds, y_true_tst_folds = tidying_preds(
            params, kf, X_trn,
            y_trn,
            stacking_regressor, trn_label)
        trn_res, tst_res = tidying_res(params, trn_dat, y_pred_trn_folds, y_prob_trn_folds, y_true_trn_folds, y_pred_tst_folds, y_prob_tst_folds, y_true_tst_folds, le,
                                       'Ensemble results-'+str(model_ensem)+'-'+pre_mode)

def tidying_preds(params, kf, X_trn, y_trn, best_model, trn_label):
    y_pred_trn_folds = []
    y_prob_trn_folds = []
    y_true_trn_folds = []
    y_pred_tst_folds = []
    y_prob_tst_folds = []
    y_true_tst_folds = []
    for train_index, test_index in kf.split(X_trn):
        X_trn_fold, X_tst_fold = X_trn[train_index], X_trn[test_index]
        y_trn_fold, y_tst_fold = y_trn[train_index], y_trn[test_index]
        best_model.fit(X_trn_fold, y_trn_fold)
        # predicted
        # prob
        y_trn_prob = best_model.predict_proba(X_trn_fold)[:, 1]
        y_test_prob = best_model.predict_proba(X_tst_fold)[:, 1]
        optimal_th, optimal_point = cal_youden(y_trn_fold, y_trn_prob)
        # pred
        y_pred_trn = [1 if val >= optimal_th else 0 for val in y_trn_prob]
        y_pred_tst = [1 if val >= optimal_th else 0 for val in y_test_prob]

        y_pred_trn_folds.append(y_pred_trn)
        y_prob_trn_folds.append(y_trn_prob)
        y_true_trn_folds.append(y_trn_fold)
        y_pred_tst_folds.append(y_pred_tst)
        y_prob_tst_folds.append(y_test_prob)
        y_true_tst_folds.append(y_tst_fold)

    return y_pred_trn_folds, y_prob_trn_folds,y_true_trn_folds, \
        y_pred_tst_folds, y_prob_tst_folds, y_true_tst_folds

def tidying_res(params, trn_dat, y_pred_trn_folds, y_prob_trn_folds, y_true_trn_folds, y_pred_tst_folds, y_prob_tst_folds, y_true_tst_folds, le,mode):
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble'))
    # predicted_values
    Folds = []
    for i in range(len(y_pred_trn_folds)):
        Fold = [i for _ in range(len(y_pred_trn_folds[i]))]
        Folds.append(Fold)
    tst_Folds = []
    for i in range(len(y_pred_tst_folds)):
        Fold = [i for _ in range(len(y_pred_tst_folds[i]))]
        tst_Folds.append(Fold)
    predictions = [item for sublist in y_pred_trn_folds for item in sublist]
    prob = [item for sublist in y_prob_trn_folds for item in sublist]
    True_Labels = [item for sublist in y_true_trn_folds for item in sublist]
    tst_predictions = [item for sublist in y_pred_tst_folds for item in sublist]
    tst_prob = [item for sublist in y_prob_tst_folds for item in sublist]
    tst_True_Labels = [item for sublist in y_true_tst_folds for item in sublist]
    Folds = [item for sublist in Folds for item in sublist]
    tst_Folds = [item for sublist in tst_Folds for item in sublist]

    trn_res = pd.DataFrame({
        'Fold': Folds,
        'Pred_label': predictions,
        'True_label': True_Labels,
        'Pred_Prob': prob
    })
    tst_res = pd.DataFrame({
        'Fold': tst_Folds,
        'Pred_label': tst_predictions,
        'True_label': tst_True_Labels,
        'Pred_Prob': tst_prob
    })
    trn_res['Pred_label'] = [le.inverse_transform([label])[0] for label in trn_res['Pred_label']]
    tst_res['Pred_label'] = [le.inverse_transform([label])[0] for label in tst_res['Pred_label']]
    trn_res['True_label'] = [le.inverse_transform([label])[0] for label in trn_res['True_label']]
    tst_res['True_label'] = [le.inverse_transform([label])[0] for label in tst_res['True_label']]

    # save results
    trn_res.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble', f'Trn dat-{mode}.csv'), index=None)
    tst_res.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble', f'Test dat-{mode}.csv'), index=None)

    return trn_res, tst_res

def ensemble_res_combined(params,Ensemble_Fileslist, Ensemble_Path):
    Ensemble_RES = pd.DataFrame()
    for Ensemble_file in Ensemble_Fileslist:
        split_0 = Ensemble_file.split('-')
        opt_pre_methods = split_0[3].split('.csv')[0]
        set = split_0[0]
        methods = split_0[2]
        # read dat and processing statistical analysis
        ensemble_df = pd.read_csv(os.path.join(Ensemble_Path, Ensemble_file))
        if params['HadTest']:
            ensemble_df.columns = ['True_label', 'Pred_label', 'Prob']
        else:
            ensemble_df.columns = ['Fold','True_label', 'Pred_label', 'Prob']
        ensemble_df['Condition'] = opt_pre_methods
        ensemble_metrics_df, ensemble_df = ml_matrics(ensemble_df)
        ensemble_metrics_df['Ensemble_Methods'] = methods
        ensemble_metrics_df['Conditions'] = ensemble_metrics_df.index
        ensemble_metrics_df['Set'] = set
        Ensemble_RES = pd.concat([Ensemble_RES, ensemble_metrics_df])
    return Ensemble_RES

def ensemble_save_res(params,Ensemble_RES):
    # save results
    trn_ensemble_res = Ensemble_RES[Ensemble_RES['Set'] == 'Trn dat']
    tst_ensemble_res = Ensemble_RES[Ensemble_RES['Set'] == 'Test dat']
    trn_ensemble_res.to_csv(
        os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble/Plotting',
                     'Ensemble Learning_performance_Train.csv'), index=None)
    tst_ensemble_res.to_csv(
        os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble/Plotting',
                     'Ensemble Learning_performance_Test.csv'), index=None)
    return trn_ensemble_res, tst_ensemble_res

def plotting_ensemble(params,trn_ensemble_res,tst_ensemble_res):
    # 计算综合性能
    trn_ensemble_res['Comprehensive Performance'] = trn_ensemble_res.iloc[:, 0:4].mean(axis=1)
    tst_ensemble_res['Comprehensive Performance'] = tst_ensemble_res.iloc[:, 0:4].mean(axis=1)

    # 合并训练和测试结果
    combined_df = pd.concat([trn_ensemble_res, tst_ensemble_res])
    combined_df = combined_df.sort_values(by='Comprehensive Performance', ascending=False)

    # 根据数据长度动态调整图的长宽
    num_rows = len(combined_df)  # 数据行数
    base_height = 8  # 基础高度
    height_per_row = 0.1 # 每行增加的高度
    figure_height = base_height + (num_rows * height_per_row)  # 动态计算总高度
    figure_width = 10  # 固定宽度

    # 设置颜色映射
    color_map = {'Trn dat': '#ffbc1f', 'Test dat': '#00509d'}

    # 创建图形
    plt.figure(figsize=(figure_width, figure_height))
    sns.set_theme(style="whitegrid")

    # 创建柱状图
    sns.barplot(
        x='Comprehensive Performance',
        y='Ensemble_Methods',
        hue='Set',
        data=combined_df,
        palette=color_map,
        dodge=True  # 分开训练和测试的柱状图
    )

    # 添加标签和标题
    plt.xlabel('Comprehensive Performance', fontsize=14)
    plt.ylabel('', fontsize=14)
    plt.legend(title='', title_fontsize=14, fontsize=14, loc='upper center', bbox_to_anchor=(0.5, 1.03), ncol=2)

    # 显示图形
    plt.tight_layout()
    plt.savefig(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Ensemble/Plotting',
                             'Ensemble Learning_performance_plot.pdf'))
    # plt.show()