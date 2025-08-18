import json
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from sklearn.inspection import permutation_importance
from sksurv.metrics import concordance_index_censored
from lifelines.exceptions import ConvergenceError
from myapp.main_processing.SA_modules.SA_Plotting import SA_Plotting, RSF_Plotting

def tidying_multi_cox_dat(params, trn_dat, trn_label, test_dat, test_label):
    trn_label.index = trn_label['samples']
    if params['HadTest']: test_label.index = test_label['samples']
    if params['SA_cofactor']:
        SA_cofactor_list = json.loads(params['SA_cofactor_list'])
    else:
        SA_cofactor_list = []
    all_Trn_dat = pd.concat([trn_label, trn_dat], axis=1)
    if params['HadTest']:
        all_Tst_dat = pd.concat([test_label, test_dat], axis=1)
    else:
        all_Tst_dat = None

    return all_Trn_dat, all_Tst_dat, SA_cofactor_list

def multi_rsf_NoTest(params, trn_dat, trn_label, test_dat, test_label,mode, model):
    trn_roc_res = None
    # processing dat
    all_Trn_dat, all_Tst_dat, SA_cofactor_list = tidying_multi_cox_dat(params, trn_dat, trn_label,
                                                                       test_dat, test_label)
    headers = SA_cofactor_list + trn_dat.columns.tolist()
    X_trn, y_trn, y_tst = rsf_input_dat(params, all_Trn_dat, all_Tst_dat, headers)

    # selecting features
    # cross-fold
    kf = KFold(n_splits=params['CV'], shuffle=True, random_state=777)
    fold_number = 0
    features_results = {}
    for train_index, test_index in kf.split(X_trn):
        train_data = X_trn[train_index]
        test_data = X_trn[test_index]
        train_y_data = y_trn[train_index]
        test_y_data = y_trn[test_index]
        # model
        features_result, best_rsf = rsf_sel_feat(params, train_data, train_y_data, headers)
        features_result['fold'] = fold_number
        features_results.update({fold_number: features_result})
        fold_number += 1
    all_features_results = pd.concat(list(features_results.values()), ignore_index=True)
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA'))
    all_features_results.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA',
                                         f'Survival analysis Multi-variable results - {model} - {mode}.csv'))
    sel_rsf_features = list(set(all_features_results['feature'][all_features_results['ranking'] == 1]))
    X_trn = all_Trn_dat[sel_rsf_features].to_numpy('float32')
    fold_number = 0
    roc_res = {}
    for train_index, test_index in kf.split(X_trn):
        train_data = X_trn[train_index]
        train_y_data = y_trn[train_index]
        # model
        features_result, best_rsf = rsf_sel_feat(params, train_data, train_y_data, headers)
        risk_train = best_rsf.predict(train_data)
        # roc
        if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA')):
            os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA'))
        trn_roc_res = RSF_Plotting(params['Parent_FilePath'], params['project_name'], train_y_data, all_Trn_dat.iloc[train_index,:], risk_train, mode, model, str('Train')+'_fold '+str(fold_number))
        roc_res.update({fold_number: trn_roc_res})
        fold_number += 1
    return [all_features_results, roc_res]

def rsf_input_dat(params, all_Trn_dat, all_Tst_dat, headers):
    X_trn = all_Trn_dat[headers].to_numpy('float32')
    y_trn = np.zeros(len(all_Trn_dat), dtype={'names': ('status', 'time'), 'formats': ('bool', 'f8')})
    y_trn['status'] = all_Trn_dat['status'].astype(bool)
    y_trn['time'] = all_Trn_dat['times']
    if params['HadTest']:
        y_tst = np.zeros(len(all_Tst_dat), dtype={'names': ('status', 'time'), 'formats': ('bool', 'f8')})
        y_tst['status'] = all_Tst_dat['status'].astype(bool)
        y_tst['time'] = all_Tst_dat['times']
    else:
        y_tst = None

    return X_trn, y_trn, y_tst

def rsf_sel_feat(params, X_trn, y_trn, headers):
    CV = params['CV']
    features_results = best_rsf = None  # 提前定义
    while CV >= 2:
        try:
            best_rsf = RandomSurvivalForest(n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1,
                                       random_state=42)
            best_rsf.fit(X_trn, y_trn)
            # 计算特征重要性
            kf = KFold(n_splits=CV, shuffle=True, random_state=42)
            feature_importance_list = []
            for train_index, test_index in kf.split(X_trn):
                x, X_tst = X_trn[train_index], X_trn[test_index]
                y, y_tst = y_trn[train_index], y_trn[test_index]
                rsf = RandomSurvivalForest(
                    n_estimators=100, min_samples_split=10, min_samples_leaf=15, n_jobs=-1, random_state=42
                )
                rsf.fit(x, y)
                # 计算排列重要性
                feature_importance = permutation_importance(
                    rsf, X_tst, y_tst, n_repeats=10, random_state=42,
                    scoring=lambda model, x, y: concordance_index_censored(
                        y['status'], y['time'], model.predict(x)
                    )[0]
                )

                feature_importance_list.append(feature_importance['importances_mean'])
            break
        except ValueError as e:
            print(f"Error with n_splits={CV}: {e}")
            CV -= 1

    # 如果 n_splits 达到 2 仍然无法满足条件，设置 features_results 为 None
    if CV < 2:
        features_results = None
        best_rsf = None
    if feature_importance_list:
        mean_feature_importance = np.mean(feature_importance_list, axis=0)
        features_results = pd.DataFrame({
            'Feature': headers,
            'RSF Importance': mean_feature_importance
        })
        features_results = features_results.sort_values(by='RSF Importance', ascending=False)

    return features_results, best_rsf

def multi_cox_NoTest(params,trn_dat, trn_label,test_dat,test_label, mode, model):
    all_Trn_dat, all_Tst_dat, SA_cofactor_list = tidying_multi_cox_dat(params, trn_dat, trn_label, test_dat, test_label)
    # initial model
    cph = CoxPHFitter(penalizer=0.1)
    formula = ' + '.join([f'`{col}`' for col in SA_cofactor_list + all_Trn_dat.columns[3:].tolist()])
    # cross-fold
    kf = KFold(n_splits=params['CV'], shuffle=True, random_state=1)
    fold_number = 0
    roc_res = {}
    for train_index, test_index in kf.split(all_Trn_dat):
        # 分割数据
        train_data = all_Trn_dat.iloc[train_index]
        test_data = all_Trn_dat.iloc[test_index]
        # 训练模型
        try:
            model = 'Successful'
            cph.fit(train_data, duration_col='times', event_col='status', formula=formula)
        except ConvergenceError:
            model = None
        if model is not None:
            # 预测测试集
            trn_pred = cph.predict_survival_function(train_data)
            if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA')):
                os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA'))
            cph.summary.to_csv(
                os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA', f'Survival analysis Multi-variable results - {model} - {mode} - fold {fold_number}.csv'))
            trn_pred.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA',
                                         f'Survival analysis Multi-variable results - Train_SA_Predictions- {model} - {mode} - fold {fold_number}.csv'))
            trn_ROC_res = SA_Plotting(params['Parent_FilePath'], params['project_name'], trn_pred,  train_data,  mode, model, str('Train')+'_fold '+str(fold_number))
            roc_res.update({fold_number:trn_ROC_res})
            fold_number += 1
    return [roc_res]

def SA_Analysis(params, mode, trn_dat, trn_label,test_dat, test_label,SA_missing, SA_scaling):
    ## Uni analysis
    trn_dat, test_dat = uni_SA_analysis(params,mode,trn_dat,trn_label,test_dat)
    # multi_cox
    multi_cox_HadTest(params, trn_dat, trn_label, test_dat, test_label, mode)
    # RSF
    multi_rsf_HadTest(params, trn_dat, trn_label, test_dat, test_label, mode)
    print('Finished Survival Analysis based on: ' + str(SA_missing) + '-' + str(SA_scaling))

def SA_Analysis_NoTest(params, mode, trn_dat, trn_label,test_dat,test_label,SA_missing, SA_scaling):
    trn_dat_uni, test_dat_uni = uni_SA_analysis(params, trn_dat, trn_dat, test_dat, mode)
    # multi_cox
    multi_cox_NoTest(params, trn_dat_uni, trn_label, test_dat_uni, test_label,mode, 'Multi_cox')
    # rf_cox
    multi_rsf_NoTest(params, trn_dat_uni, trn_label, test_dat_uni, test_label,mode, 'RSF')
    print('Finished Survival Analysis based on: ' + str(SA_missing) + '-' + str(SA_scaling))

def uni_SA_analysis(params,mode,trn_dat,trn_label,test_dat):
    if os.path.exists(
            os.path.join(params['Parent_FilePath'], params['project_name'], 'Results',
                         f'Survival analysis Univariable results - {mode}.csv')):
        trn_dat_trans = trn_dat.transpose()
        if params['HadTest']: test_dat_trans = test_dat.transpose()
        if params['LoadUni']:
            results_df = pd.read_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results',
                                                  f'Survival analysis Univariable results - {mode}.csv'),
                                     index_col=0)
            Sig_var = list(results_df.index[results_df['P'] < 0.05])
            if len(Sig_var) > 0:
                trn_dat = trn_dat_trans[Sig_var]
                if params['HadTest']:
                    test_dat = test_dat_trans[Sig_var]
                else:
                    test_dat = None
            else:
                Ex_var = results_df.sort_values(by='P', ascending=True).head(100).index
                trn_dat = trn_dat_trans[Ex_var]
                if params['HadTest']:
                    test_dat = test_dat_trans[Ex_var]
                else:
                    test_dat = None
                print('No significant variables in survival analysis after uni-variables analysis')
    else:
        trn_dat_trans = trn_dat.transpose()
        if params['HadTest']: test_dat_trans = test_dat.transpose()
        if params['SA_cofactor']:
            SA_cofactor_list = json.loads(params['SA_cofactor_list'])
        else:
            SA_cofactor_list = []
        results = {}
        for column in tqdm(trn_dat_trans.columns):
            attempts = 0
            success = False
            while attempts < 3 and not success:
                try:
                    T = trn_label[['times', 'group'] + SA_cofactor_list].copy()
                    T[[column]] = trn_dat_trans[[column]].values
                    # 初始化 CoxPHFitter 并拟合模型
                    cph = CoxPHFitter()
                    formula = ' + '.join(
                        [f'`{col}`' if '-' in col else col for col in SA_cofactor_list + [column]])
                    cph.fit(T, duration_col='times', event_col='group', formula=formula)
                    # 提取结果
                    results.update({column: {
                        'HR': cph.summary.loc[column, 'exp(coef)'],
                        'HR_lower': cph.summary.loc[column, 'exp(coef) lower 95%'],
                        'HR_upper': cph.summary.loc[column, 'exp(coef) upper 95%'],
                        'P': cph.summary.loc[column, 'p']
                    }})
                    success = True
                except Exception as e:
                    attempts += 1
                    if attempts == 3:
                        print(f"Error processing column '{column}': {e}")
                        results.update({column: {
                            'HR': None,
                            'HR_lower': None,
                            'HR_upper': None,
                            'P': None
                        }})
        if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results')):
            os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results'))
        results_df = pd.DataFrame(results).transpose()
        results_df.to_csv(
            os.path.join(params['Parent_FilePath'], params['project_name'], 'Results',
                         f'Survival analysis Univariable results - {mode}.csv'))
        if params['LoadUni']:
            Sig_var = list(results_df.index[results_df['P'] < 0.05])
            if len(Sig_var) > 0:
                trn_dat = trn_dat_trans[Sig_var]
                if params['HadTest']:
                    test_dat = test_dat_trans[Sig_var]
                else:
                    test_dat = None
            else:
                Ex_var = results_df.sort_values(by='P', ascending=True).head(100).index
                trn_dat = trn_dat_trans[Ex_var]
                if params['HadTest']:
                    test_dat = test_dat_trans[Ex_var]
                else:
                    test_dat = None
                print('No significant variables in survival analysis after uni-variables analysis')
    return trn_dat,test_dat

def multi_cox_HadTest(params, trn_dat, trn_label, test_dat, test_label,mode):
    model = 'Multi-Cox'
    all_Trn_dat, all_Tst_dat, SA_cofactor_list = tidying_multi_cox_dat(params, trn_dat, trn_label, test_dat, test_label)
    # initial model
    cph = CoxPHFitter(penalizer=0.1)
    formula = ' + '.join([f'`{col}`' for col in SA_cofactor_list + all_Trn_dat.columns[3:].tolist()])
    # 拟合 Cox 模型
    cph.fit(all_Trn_dat, duration_col='times', event_col='group', formula=formula, robust=True)
    # validated model performance
    trn_pred = cph.predict_survival_function(all_Trn_dat)
    tst_pred = cph.predict_survival_function(all_Tst_dat)
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA'))
    cph.summary.to_csv(
        os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA',
                     f'Survival analysis Multi-variable results - {model} - {mode}.csv'))
    trn_pred.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA',
                                 f'Survival analysis Multi-variable results - Train_SA_Predictions - {model} - {mode}.csv'))
    tst_pred.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA',
                                 f'Survival analysis Multi-variable results - Test_SA_Predictions - {model} - {mode}.csv'))
    SA_Plotting(params, trn_pred, all_Trn_dat, mode, model, 'Train')
    SA_Plotting(params, tst_pred, all_Tst_dat, mode, model, 'Test')

def multi_rsf_HadTest(params, trn_dat, trn_label,test_dat, test_label,mode):
    model = 'RSF'
    # processing dat
    all_Trn_dat, all_Tst_dat, SA_cofactor_list = tidying_multi_cox_dat(params, trn_dat, trn_label,
                                                                       test_dat, test_label)
    headers = SA_cofactor_list + trn_dat.columns.tolist()
    X_trn, y_trn, y_tst = rsf_input_dat(params, all_Trn_dat, all_Tst_dat, headers)
    # selecting features
    features_results, best_rsf = rsf_sel_feat(params, X_trn, y_trn, headers)
    if best_rsf is not None:
        sel_rsf_features = features_results['Feature'][features_results['RSF Importance'] != 0].tolist()
        # re-constructed model
        X_trn_sel = all_Trn_dat[sel_rsf_features].to_numpy('float32')
        X_tst_sel = all_Tst_dat[sel_rsf_features].to_numpy('float32')
        headers_sel = [header for header in headers if header in sel_rsf_features]
        features_results_sel, best_rsf = rsf_sel_feat(params, X_trn_sel, y_trn, headers_sel)
        risk_train = best_rsf.predict(X_trn_sel)
        risk_tst = best_rsf.predict(X_tst_sel)
        # roc
        if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA')):
            os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA'))
        features_results.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA',
                                             f'Survival analysis Multi-variable results - {model} - {mode}.csv'))
        pd.DataFrame(risk_train).to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA',
                                                     f'Survival analysis Multi-variable results - Train_SA_Predictions - {model} - {mode}.csv'))
        pd.DataFrame(risk_tst).to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/SA',
                                                   f'Survival analysis Multi-variable results - Test_SA_Predictions - {model} - {mode}.csv'))
        RSF_Plotting(params, y_trn, all_Trn_dat, risk_train, mode, model, 'Train')
        RSF_Plotting(params, y_tst, all_Tst_dat, risk_tst, mode, model, 'Test')


