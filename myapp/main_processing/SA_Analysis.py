import os
import pandas as pd

from myapp.tools.utils import tidy_missing_mean, tidy_missing_MI, tidy_missing_KNN, scaling_dat
from myapp.main_processing.ML_modules.ML_processing import *

def SA_data_processing(params, trn_dat, test_dat, missing_ratio, missing_test_ratio):
    ## prepared processing methods
    pre_missings = ['Mean','MI', 'KNN']
    pre_scalings = ['None', 'MinMax', 'Zscore', 'lg', 'RobustScaler', 'MaxAbs', 'PowerTransformer']
    # train dat
    if missing_ratio != 0:
        for tidymiss in pre_missings:
            if tidymiss == 'Mean':
                trn_dat_missing = tidy_missing_mean(trn_dat)
            elif tidymiss == 'MI':
                trn_dat_missing = tidy_missing_MI(trn_dat)
            elif tidymiss == 'KNN':
                trn_dat_missing = tidy_missing_KNN(trn_dat)
            else:
                raise ValueError("Invalid tidymiss value. Please choose 'Mean', 'MI', or 'KNN'.")
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
    return pre_missings, pre_scalings

def scaling_dat_for(params, dat, pre_scalings, mode, tidymiss):
    for scaling in pre_scalings:
        dat_scale = scaling_dat(scaling, dat)
        # save data
        if not os.path.exists(
                os.path.join(params['Parent_FilePath'], params['project_name'], 'Data_Processing')):
            os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Data_Processing'))
        dat_scale.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Data_Processing',
                                      f'{mode}_dat_after_{tidymiss}-{scaling}.csv'))