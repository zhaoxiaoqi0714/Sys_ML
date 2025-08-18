import os
import numpy as np
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.preprocessing import MinMaxScaler,StandardScaler, RobustScaler, MaxAbsScaler, PowerTransformer
from imblearn.over_sampling import SMOTE, ADASYN,BorderlineSMOTE,RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTEENN,SMOTETomek
from collections import Counter

def load_org_dat(params):
    if params['HadLabel']:
        if params['HadTest']:
            trn_dat = pd.read_csv(os.path.join(params['Parent_FilePath'],params['project_name'], 'train_matrix.csv'), index_col=0)
            test_dat = pd.read_csv(os.path.join(params['Parent_FilePath'],params['project_name'], 'test_matrix.csv'), index_col=0)
            trn_label = pd.read_csv(os.path.join(params['Parent_FilePath'],params['project_name'], 'train_group.csv'))
            test_label = pd.read_csv(os.path.join(params['Parent_FilePath'],params['project_name'], 'test_group.csv'))
        else:
            trn_dat = pd.read_csv(os.path.join(params['Parent_FilePath'],params['project_name'], 'train_matrix.csv'), index_col=0)
            trn_label = pd.read_csv(os.path.join(params['Parent_FilePath'],params['project_name'], 'train_group.csv'))
            test_dat = None
            test_label = None
    else:
        if params['HadTest']:
            trn_dat = pd.read_csv(os.path.join(params['Parent_FilePath'],params['project_name'], 'train_matrix.csv'), index_col=0)
            test_dat = pd.read_csv(os.path.join(params['Parent_FilePath'],params['project_name'], 'test_matrix.csv'), index_col=0)
            trn_label = None
            test_label = None
        else:
            trn_dat = pd.read_csv(os.path.join(params['Parent_FilePath'],params['project_name'], 'train_matrix.csv'), index_col=0)
            trn_label = None
            test_dat = None
            test_label = None
    return trn_dat, test_dat, trn_label, test_label

def tidy_missing_mean(dat):
    dat.fillna(dat.mean(), inplace=True)
    assert dat.isna().any().any() == False, 'Missing value still exists in matrix.'

    return dat

def tidy_missing_MI(dat):
    max_iter = 10
    imputer = IterativeImputer(max_iter=max_iter, random_state=777)
    dat = pd.DataFrame(imputer.fit_transform(dat), columns=dat.columns, index=dat.index)
    dat = dat.clip(lower=0)
    assert dat.isna().any().any() == False, 'Missing value still exists in matrix.'

    return dat

def tidy_missing_KNN(dat):
    imputer = KNNImputer(n_neighbors=5)
    dat_imputed = imputer.fit_transform(dat)
    dat = pd.DataFrame(dat_imputed, columns=dat.columns, index=dat.index)
    assert dat.isna().any().any() == False, 'Missing value still exists in matrix.'

    return dat

def tidying_missing_test(test_dat, tidymiss):
    if tidymiss == 'Mean':
        test_dat = tidy_missing_mean(test_dat)
    elif tidymiss == 'MI':
        test_dat = tidy_missing_MI(test_dat)
    elif tidymiss == 'KNN':
        test_dat = tidy_missing_KNN(test_dat)
    elif tidymiss == 'None':
        test_dat = test_dat
    else:
        raise ValueError("Invalid tidymiss value. Please choose 'Mean', 'MI', or 'KNN'.")

    return test_dat

def scaling_dat(scaling, dat):
    if scaling == 'MinMax':
        scaler = MinMaxScaler()
        dat_scaled = scaler.fit_transform(dat)
        dat = pd.DataFrame(dat_scaled, columns=dat.columns, index=dat.index)
    elif scaling == 'None':
        dat = dat
    elif scaling == 'Zscore':
        scaler = StandardScaler()
        dat_scaled = scaler.fit_transform(dat)
        dat = pd.DataFrame(dat_scaled, columns=dat.columns, index=dat.index)
    elif scaling == 'lg':
        if (dat < 0).any().any():  # 适用于DataFrame和array
            print("Warning: Negative values detected - skipping log transformation")
            return dat.copy()  # 返回原始数据的副本
        else:
            return np.log10(dat.astype(float) + 0.001)  # 避免对0取对数
    elif scaling == 'RobustScaler':
        scaler = RobustScaler()
        dat_scaled = scaler.fit_transform(dat)
        dat = pd.DataFrame(dat_scaled, columns=dat.columns, index=dat.index)
    elif scaling == 'MaxAbs':
        scaler = MaxAbsScaler()
        dat_scaled = scaler.fit_transform(dat)
        dat = pd.DataFrame(dat_scaled, columns=dat.columns, index=dat.index)
    elif scaling == 'PowerTransformer':
        scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        dat_scaled = scaler.fit_transform(dat)
        dat = pd.DataFrame(dat_scaled, columns=dat.columns, index=dat.index)
    else:
        raise ValueError("Invalid ScalingMethods value. Please choose 'MinMax', 'Zscore', 'RobustScaler', 'MaxAbs', 'PowerTransformer' or 'lg'.")

    return dat

def proc_Imb(imb, X, y):
    print('Dataset shape %s' % Counter(y))
    if imb == 'ADASYN':
        adasyn = ADASYN(random_state=777)
        X_resampled, y_resampled = adasyn.fit_resample(X, y)
    elif imb == 'None':
        X_resampled, y_resampled = X, y
    elif imb == 'Borderline-SMOTE':
        sm = BorderlineSMOTE(random_state=777, kind="borderline-1")
        X_resampled, y_resampled = sm.fit_resample(X, y)
    elif imb == 'RandomOverSample':
        ros = RandomOverSampler()
        X_resampled, y_resampled = ros.fit_resample(X, y)
    elif imb == 'RandomUnderSampler':
        rus = RandomUnderSampler()
        X_resampled, y_resampled = rus.fit_resample(X, y)
    elif imb == 'TomekLinks':
        tl = TomekLinks()
        X_resampled, y_resampled = tl.fit_resample(X, y)
    elif imb == 'SMOTE':
        smote = SMOTE(sampling_strategy='auto', random_state=777)
        X_resampled, y_resampled = smote.fit_resample(X, y)
    elif imb == 'SMOTEENN':
        smoteenn = SMOTEENN()
        X_resampled, y_resampled = smoteenn.fit_resample(X, y)
    elif imb == 'SMOTETomek':
        smotetomek = SMOTETomek()
        X_resampled, y_resampled = smotetomek.fit_resample(X, y)
    else:
        raise ValueError("Invalid processing imbalance label value.")
    print(f'After processing imbalance data, Methods: {imb}, Dataset shape %s' % Counter(y_resampled))

    return X_resampled, y_resampled
