import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from myapp.tools.ML_RES import load_ML_res, tidying_ML_res, ml_matrics,Sys_ML_matrics_plotting,tidying_feats_res,Sys_ML_feats_ploting
from myapp.main_processing.ML_Analysis import Uni_analysis, processing_ML, processing_ML_CV
from myapp.tools.utils import tidy_missing_mean, tidy_missing_MI, tidy_missing_KNN, scaling_dat,proc_Imb
from myapp.main_processing.ML_modules.Unsup_processing import *
from myapp.main_processing.SA_modules.SA_Processing import SA_Analysis,SA_Analysis_NoTest

def MLs_pipelines(params, MLs_list,Mls_Recommend,missing_ratio,missing_test_ratio,
                  trn_label,test_label):
    for ML in MLs_list:
        ML_missings = Mls_Recommend[ML]['Missing']
        if missing_ratio == 0 or missing_test_ratio == 0: ML_missings += ['None']
        ML_scalings = Mls_Recommend[ML]['Scaling']
        for ML_missing in ML_missings:
            for ML_scaling in ML_scalings:
                try:
                    trn_dat, test_dat,X_tst = load_ML_pre_dat(params, ML_missing, ML_scaling)
                except FileNotFoundError:
                    print(
                        f"Not File: {os.path.join(params['Parent_FilePath'], params['project_name'],'Data_Processing', f'Train_dat_after_{ML_missing}-{ML_scaling}.csv')}")
                    continue
                if params['HadLabel']:
                    ## performing uni-variables analysis
                    trn_dat, test_dat, X_trn, X_tst, y_trn,y_tst,le = data_preocessing_in_Uni(params,trn_dat,test_dat,X_tst,trn_label,test_label,ML_missing, ML_scaling)
                    # unsup analysis
                    if params['Unsup_analysis']:
                        Unsup_hadTest(params, X_trn, trn_label, ML_scaling, ML_missing, 'train')
                        if params['HadTest']: Unsup_hadTest(params, X_tst, test_label, ML_scaling,ML_missing, 'test')
                    # processing imbalance
                    if params['Imbalance']:
                        try:
                            for imb in params['imb_methods']:
                                mode = str(ML_missing) + '_' + str(ML_scaling) + '_' + imb
                                X_trn, y_trn = proc_Imb(imb, X_trn, y_trn)
                                if params['HadTest']:
                                    processing_ML(params, ML, trn_dat, X_trn, y_trn, X_tst, test_label,y_tst, le, mode)
                                else:
                                    processing_ML_CV(params, ML, trn_dat, X_trn, y_trn, le, mode)
                        except ValueError as e:
                            print(f"Error processing project: {e}")
                            continue
                    else:
                        mode = str(ML_missing) + '_' + str(ML_scaling) + '_' + 'None'
                        if params['HadTest']:
                            processing_ML(params, ML, trn_dat, X_trn, y_trn, X_tst, test_label, y_tst, le, mode)
                        else:
                            processing_ML_CV(params, ML, trn_dat, X_trn, y_trn, le, mode)
                else:
                    X_trn = trn_dat.transpose().values
                    if params['HadTest']: X_tst = test_dat.transpose().values
                    Unsup_Nolabel(params, X_trn, ML_scaling, ML_missing, 'train')
                    if params['HadTest']: Unsup_Nolabel(params, X_tst, ML_scaling, ML_missing, 'test')

def SAs_pipelines(params, missing_ratio, missing_test_ratio, pre_missings, pre_scalings,
                  trn_label,test_label):
    if params['HadLabel']:
        if missing_ratio == 0 or missing_test_ratio == 0: pre_missings += ['None']
        for SA_missing in pre_missings:
            for SA_scaling in pre_scalings:
                mode = str(SA_missing) + '-' + str(SA_scaling)
                try:
                    trn_dat,test_dat,trn_label,test_label = load_SA_pre_dat(params, SA_missing, SA_scaling, trn_label,test_label)
                except FileNotFoundError:
                    print(
                        f"Not File: {os.path.join(params['Parent_FilePath'], params['project_name'], 'Data_Processing', f'Train_dat_after_{SA_missing}-{SA_scaling}.csv')}")
                    continue
                if params['HadTest']:
                    SA_Analysis(params, mode, trn_dat, trn_label,test_dat, test_label,SA_missing, SA_scaling)
                else:
                    SA_Analysis_NoTest(params, mode, trn_dat, trn_label,test_dat,test_label,SA_missing, SA_scaling)
    else:
        raise ValueError("Label files are essential for conducting survival analysis.")

def MLs_res_ana(params,imb_methods):
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML/Plotting')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML/Plotting'))
    # load_ml_res
    features_scores_files, test_data_files, train_data_files = load_ML_res(params)
    final_df, trn_df, tst_df = tidying_ML_res(params, features_scores_files, test_data_files, train_data_files)
    ## analysis prediction results
    trn_metrics_df, trn_df = ml_matrics(trn_df)
    tst_metrics_df, tst_df = ml_matrics(tst_df)
    # prepared plotting df
    # trn_df
    Sys_ML_matrics_plotting(params, trn_metrics_df, imb_methods, 'Train')
    Sys_ML_matrics_plotting(params, tst_metrics_df, imb_methods, 'Test')
    trn_metrics_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML/Plotting/ML_performance_Train.csv'))
    tst_metrics_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML/Plotting/ML_performance_Test.csv'))

    ## analysis features results
    # calcularted feats correlation between different methods
    results_df = tidying_feats_res(params, final_df)
    # plotting
    Sys_ML_feats_ploting(params, results_df)

def load_ML_pre_dat(params, ML_missing, ML_scaling):
    # load dat
    test_dat = X_tst = None
    trn_dat = pd.read_csv(os.path.join(params['Parent_FilePath'], params['project_name'],
                                       'Data_Processing', f'Train_dat_after_{ML_missing}-{ML_scaling}.csv'),
                          index_col=0)
    if params['HadTest']: test_dat = pd.read_csv(
        os.path.join(params['Parent_FilePath'], params['project_name'],
                     'Data_Processing', f'Test_dat_after_{ML_missing}-{ML_scaling}.csv'), index_col=0)
    return trn_dat, test_dat,X_tst

def data_preocessing_in_Uni(params,trn_dat,test_dat,X_tst,trn_label,test_label,ML_missing, ML_scaling):
    y_tst = None
    ## performing uni-variables analysis
    trn_dat, test_dat = Uni_analysis(params, trn_dat, trn_label, test_dat,
                                     ML_missing + '-' + ML_scaling)
    ## processing data
    le = LabelEncoder()
    trn_label['group_encoded'] = le.fit_transform(trn_label['group'])
    X_trn = trn_dat.values
    y_trn = trn_label['group_encoded'].values
    if params['HadTest']:
        test_label['group_encoded'] = le.transform(test_label['group'])
        X_tst = test_dat.values
        y_tst = test_label['group_encoded'].values
    return trn_dat, test_dat, X_trn, X_tst, y_trn,y_tst,le

def load_SA_pre_dat(params, SA_missing, SA_scaling,trn_label,test_label):
    # load dat
    trn_dat = pd.read_csv(os.path.join(params['Parent_FilePath'], params['project_name'],
                                       'Data_Processing', f'Train_dat_after_{SA_missing}-{SA_scaling}.csv'),
                          index_col=0)
    if params['HadTest']: test_dat = pd.read_csv(
        os.path.join(params['Parent_FilePath'], params['project_name'],
                     'Data_Processing', f'Test_dat_after_{SA_missing}-{SA_scaling}.csv'), index_col=0)
    trn_label, test_label = SA_label_tidy(params,SA_missing,trn_label,test_label)
    trn_label_scale = scaling_col_dat(trn_label, SA_scaling)
    if params['HadTest']: test_label_scale = scaling_col_dat(test_label, SA_scaling)

    return trn_dat,test_dat,trn_label_scale,test_label_scale

def SA_label_tidy(params,SA_missing,trn_label,test_label):
    if SA_missing is not None:
        if trn_label.isna().any().any():
            numeric_cols = trn_label.select_dtypes(include=[np.number]).columns
            if SA_missing == 'Mean':
                trn_label[numeric_cols] = tidy_missing_mean(trn_label[numeric_cols])
            elif SA_missing == 'MI':
                trn_label[numeric_cols] = tidy_missing_MI(trn_label[numeric_cols])
            elif SA_missing == 'KNN':
                trn_label[numeric_cols] = tidy_missing_KNN(trn_label[numeric_cols])
            elif SA_missing == 'None':
                trn_label[numeric_cols] = tidy_missing_mean(trn_label[numeric_cols])
            else:
                raise ValueError("Invalid tidymiss value. Please choose 'Mean', 'MI', or 'KNN'.")
        if params['HadTest']:
            if test_label.isna().any().any():
                numeric_cols = test_label.select_dtypes(include=[np.number]).columns
                if SA_missing == 'Mean':
                    test_label[numeric_cols] = tidy_missing_mean(test_label[numeric_cols])
                elif SA_missing == 'MI':
                    test_label[numeric_cols] = tidy_missing_MI(test_label[numeric_cols])
                elif SA_missing == 'KNN':
                    test_label[numeric_cols] = tidy_missing_KNN(test_label[numeric_cols])
                elif SA_missing == 'None':
                    test_label[numeric_cols] = tidy_missing_mean(test_label[numeric_cols])
                else:
                    raise ValueError("Invalid tidymiss value. Please choose 'Mean', 'MI', or 'KNN'.")
    return trn_label,test_label

def is_non_integer(series):
    return series.apply(lambda x: not np.isclose(x, np.round(x)))

def scaling_col_dat(dat, scaling):
    non_integer_cols = dat.select_dtypes(include=[np.number]).columns[
        dat.select_dtypes(include=[np.number]).apply(is_non_integer).any()
    ]
    non_integer_cols = [col for col in non_integer_cols if col != 'times']
    for col in non_integer_cols:
        col_data = dat[[col]]
        scaled_col_data = scaling_dat(scaling, col_data)
        dat[col] = scaled_col_data[col]
    return dat
