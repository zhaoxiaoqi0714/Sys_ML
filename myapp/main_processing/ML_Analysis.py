import os
import pandas as pd
import time
from datetime import datetime
from scipy.stats import mannwhitneyu, kruskal

from myapp.tools.utils import tidy_missing_mean, tidy_missing_MI, tidy_missing_KNN, scaling_dat
from myapp.main_processing.ML_modules.ML_processing import *

def ML_data_processing(params, trn_dat, test_dat, MLs_list, Mls_Recommend, missing_ratio, missing_test_ratio):
    ## prepared processing methods
    pre_missings = []
    pre_scalings = []
    pre_imbalances = []
    for ML in MLs_list:
        pre_missings.append([miss for miss in Mls_Recommend[ML]['Missing']])
        pre_scalings.append([miss for miss in Mls_Recommend[ML]['Scaling']])
        pre_imbalances.append([miss for miss in Mls_Recommend[ML]['imb_methods']])

    pre_missings = list(set([item for sublist in pre_missings for item in sublist]))
    pre_scalings = list(set([item for sublist in pre_scalings for item in sublist]))
    pre_imbalances = list(set([item for sublist in pre_imbalances for item in sublist]))

    # train dat
    if missing_ratio != 0:
        for tidymiss in pre_missings:
            start_time = time.time()  # 记录开始时间
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            if tidymiss == 'Mean':
                trn_dat_missing = tidy_missing_mean(trn_dat)
            elif tidymiss == 'MI':
                trn_dat_missing = tidy_missing_MI(trn_dat)
            elif tidymiss == 'KNN':
                trn_dat_missing = tidy_missing_KNN(trn_dat)
            else:
                raise ValueError("Invalid tidymiss value. Please choose 'Mean', 'MI', or 'KNN'.")
            elapsed_time = time.time() - start_time
            print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {tidymiss} Finished, Elapsed_time: {elapsed_time:.2f} s")
            ## processing scalings
            scaling_dat_for(params, trn_dat_missing, pre_scalings, 'Train', tidymiss)
    else:
        tidymiss = 'None'
        scaling_dat_for(params, trn_dat, pre_scalings, 'Train', tidymiss)

    # test dat
    if params['HadTest']:
        if missing_test_ratio != 0:
            for tidymiss in pre_missings:
                if tidymiss == 'Mean':
                    test_dat_missing = tidy_missing_mean(test_dat)
                elif tidymiss == 'MI':
                    test_dat_missing = tidy_missing_MI(test_dat)
                elif tidymiss == 'KNN':
                    test_dat_missing = tidy_missing_KNN(test_dat)
                else:
                    raise ValueError("Invalid tidymiss value. Please choose 'Mean', 'MI', or 'KNN'.")
                ## processing scalings
                scaling_dat_for(params, test_dat_missing, pre_scalings, 'Test', tidymiss)
        else:
            tidymiss = 'None'
            scaling_dat_for(params, test_dat, pre_scalings, 'Test', tidymiss)

def scaling_dat_for(params, dat, pre_scalings, mode, tidymiss):
    for scaling in pre_scalings:
        start_time = time.time()  # Record start time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        dat_scale = scaling_dat(scaling, dat)
        elapsed_time = time.time() - start_time
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {scaling} scaling completed. Time elapsed: {elapsed_time:.2f} seconds")
        # save data
        if not os.path.exists(
                os.path.join(params['Parent_FilePath'], params['project_name'], 'Data_Processing')):
            os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Data_Processing'))
        dat_scale.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Data_Processing',
                                      f'{mode}_dat_after_{tidymiss}-{scaling}.csv'))

def Uni_analysis(params, trn_dat, trn_label, test_dat, mode):
    # trn_dat = trn_dat.transpose()
    group_counts = trn_label['group'].value_counts()
    num_groups = len(group_counts)
    if num_groups == 2:
        # Mann-Whitney U test
        results = {}
        for column in trn_dat.columns:
            group1_idx = trn_label[trn_label['group'] == group_counts.index[0]]['samples']
            group2_idx = trn_label[trn_label['group'] == group_counts.index[1]]['samples']
            u_statistic, p_value = mannwhitneyu(trn_dat.loc[group1_idx, column],
                                                trn_dat.loc[group2_idx, column],
                                                alternative='two-sided')
            results[column] = {'U_statistic': u_statistic, 'P_value': p_value}
    else:
        # Kruskal-Wallis test
        results = {}
        for column in trn_dat.columns:
            group_data = [trn_dat.loc[trn_label[trn_label['group'] == group]['samples'], column] for group
                          in group_counts.index]
            k_statistic, p_value = kruskal(*group_data)
            results[column] = {'K_statistic': k_statistic, 'P_value': p_value}
    uni_res = pd.DataFrame(results).T
    # saving results
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Uni')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Uni'))
    uni_res.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Uni', f'Univariable results-{mode}.csv'))
    if params['LoadUni']:
        Sig_var = list(uni_res.index[uni_res['P_value'] < 0.05])
        if len(Sig_var) > 0:
            trn_dat = trn_dat.loc[:, Sig_var]
            if params['HadTest']:
                test_dat = test_dat.loc[:, Sig_var]
            else:
                test_dat = None
        else:
            trn_dat = trn_dat
            test_dat = test_dat
            print('No significant variables after uni-variables analysis')

    return trn_dat,test_dat


def processing_ML(params, ML, trn_dat, X_trn, y_trn, X_tst, test_label, y_tst, le, mode):
    start_time = time.time()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        print(f"\n[{current_time}] Starting {ML} model training...")

        if ML == 'Boruta':
            Boruta_ML(params, trn_dat, X_trn, y_trn, X_tst, test_label, y_tst, le, mode)
            print('Finished Boruta')
        elif ML == 'GaussianNB':
            GaussianNB_ML(params, trn_dat, X_trn, y_trn, X_tst, test_label, y_tst, le, mode)
            print('Finished GaussianNB')
        elif ML == 'GBDT':
            GBDT_ML(params, trn_dat, X_trn, y_trn, X_tst, test_label, y_tst, le, mode)
            print('Finished GBDT')
        elif ML == 'LASSO':
            LASSO_ML(params, trn_dat, X_trn, y_trn, X_tst, test_label, y_tst, le, mode)
            # print('Finished LASSO')
        elif ML == 'Logit':
            if y_trn.max() + 1 == 2:
                Logit_ML(params, trn_dat, X_trn, y_trn, X_tst, test_label, y_tst, le, mode)
                print('Finished Logit')
            else:
                print(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Skipped Logit (binary classifier with multiclass data)")
                return
            # print('Finished Logit')
        elif ML == 'NeuralNetwork':
            NN_ML(params, trn_dat, X_trn, y_trn, X_tst, test_label, y_tst, le, mode)
            print('Finished NeuralNetwork')
        elif ML == 'PLSDA':
            PLSDA_ML(params, trn_dat, X_trn, y_trn, X_tst, test_label, y_tst, le, mode)
            print('Finished PLSDA')
        elif ML == 'RandomForest':
            RF_ML(params, trn_dat, X_trn, y_trn, X_tst, test_label, y_tst, le, mode)
            print('Finished RandomForest')
        elif ML == 'Xgboost':
            Xgboost_ML(params, trn_dat, X_trn, y_trn, X_tst, test_label, y_tst, le, mode)
            print('Finished Xgboost')
        else:
            raise ValueError(
                "Invalid Machine learning methods. Please choose 'Boruta', 'GaussianNB', 'GBDT', 'LASSO', 'Logit', "
                "'NeuralNetwork','PLSDA', 'RandomForest' or 'Xgboost'.")

        elapsed_time = time.time() - start_time
        print(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {ML} completed. Time elapsed: {elapsed_time:.2f} seconds")

    except Exception as e:
        error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{error_time}] ERROR: {ML} processing failed")
        print(f"Error details: {str(e)}")
        raise  # Re-raise the exception after logging


def processing_ML_CV(params, ML, trn_dat, X_trn, y_trn, le, mode):
    start_time = time.time()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        print(f"\n[{current_time}] Starting {ML} cross-validation...")

        # ## option SVM test
        # SVM_ML_CV(params, trn_dat, X_trn, y_trn, le, mode)
        # elapsed_time = time.time() - start_time
        # hours, rem = divmod(elapsed_time, 3600)
        # minutes, seconds = divmod(rem, 60)
        # time_str = "{:0>2}h {:0>2}m {:05.2f}s".format(int(hours), int(minutes), seconds)
        #
        # print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {ML} CV completed. Total duration: {time_str}")
        # print(f"{'=' * 60}")


        if ML == 'Boruta':
            Boruta_ML_CV(params, trn_dat, X_trn, y_trn, le, mode)
        elif ML == 'GaussianNB':
            GaussianNB_ML_CV(params, trn_dat, X_trn, y_trn, le, mode)
            # print("GaussianNB")
        elif ML == 'GBDT':
            GBDT_ML_CV(params, trn_dat, X_trn, y_trn, le, mode)
            # print("GBDT")
        elif ML == 'LASSO':
            LASSO_ML_CV(params, trn_dat, X_trn, y_trn, le, mode)
            # print("LASSO")
        elif ML == 'Logit':
            Logit_ML_CV(params, trn_dat, X_trn, y_trn, le, mode)
            # print("Logit")
        elif ML == 'NeuralNetwork':
            NN_ML_CV(params, trn_dat, X_trn, y_trn, le, mode)
            # print("NeuralNetwork")
        elif ML == 'PLSDA':
            PLSDA_ML_CV(params, trn_dat, X_trn, y_trn, le, mode)
            # print("PLSDA")
        elif ML == 'RandomForest':
            RF_ML_CV(params, trn_dat, X_trn, y_trn, le, mode)
            # print("RandomForest")
        elif ML == 'Xgboost':
            Xgboost_ML_CV(params, trn_dat, X_trn, y_trn, le, mode)
            # print("Xgboost")
        else:
            raise ValueError(
                "Invalid Machine learning method. Please choose from: 'Boruta', 'GaussianNB', 'GBDT', 'LASSO', 'Logit', "
                "'NeuralNetwork', 'PLSDA', 'RandomForest', or 'Xgboost'.")

        elapsed_time = time.time() - start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        time_str = "{:0>2}h {:0>2}m {:05.2f}s".format(int(hours), int(minutes), seconds)

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {ML} CV completed. Total duration: {time_str}")
        print(f"{'=' * 60}")

    except Exception as e:
        error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n[{error_time}] ERROR: {ML} cross-validation failed")
        print(f"Error details: {str(e)}")
        print(f"{'=' * 60}")
        raise  # Re-raise exception after logging
