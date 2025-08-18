import os
import json

from myapp.tools.utils import load_org_dat

def load_global_params(Parent_FilePath, FilePath):
    # load params files
    with open(os.path.join(FilePath, 'project_params.json'), 'r', encoding='utf-8') as file:
        params = json.load(file)

    params['Parent_FilePath'] = Parent_FilePath
    params['FilePath'] = FilePath
    tidymiss = 'None'
    tidymiss_methods = ['MI', 'KNN']
    scaling_methods = ['MinMax', 'Zscore','RobustScaler']
    imb_methods = ['None', 'ADASYN', 'Borderline-SMOTE', 'RandomOverSample', 'RandomUnderSampler', 'TomekLinks', 'SMOTE', 'SMOTETomek']
    Mls_Recommend = {
        'Boruta': {
            'Missing': ['KNN'],
            'Scaling': scaling_methods,
            'imb_methods': ['SMOTETomek']
        },
        'GaussianNB': {
            'Missing': ['MI', 'KNN'],
            'Scaling': scaling_methods,
            'imb_methods': ['SMOTETomek']
        },
        'GBDT': {
            'Missing': ['MI'],
            'Scaling': scaling_methods,
            'imb_methods': ['SMOTETomek']
        },
        'LASSO': {
            'Missing': ['Mean', 'MI', 'KNN'],
            'Scaling': scaling_methods,
            'imb_methods': ['SMOTETomek']
        },
        'Logit': {
            'Missing': ['MI', 'KNN'],
            'Scaling': scaling_methods,
            'imb_methods': ['SMOTETomek']
        },
        'NeuralNetwork': {
            'Missing': ['Mean', 'MI', 'KNN'],
            'Scaling': scaling_methods,
            'imb_methods': ['SMOTETomek']
        },
        'PLSDA': {
            'Missing': ['Mean', 'MI', 'KNN'],
            'Scaling': scaling_methods,
            'imb_methods': ['SMOTETomek']
        },
        'RandomForest': {
            'Missing': ['Mean', 'MI', 'KNN'],
            'Scaling': scaling_methods,
            'imb_methods': ['SMOTETomek']
        },
        'Xgboost': {
            'Missing': ['Mean', 'MI', 'KNN'],
            'Scaling': scaling_methods,
            'imb_methods': ['SMOTETomek']
        }
    }
    # load matrix and group files
    trn_dat = test_dat = trn_label = test_label = None
    if params['MLAnalysisMLAnalysis'] | params['SurvivalAnalysis']: trn_dat, test_dat, trn_label, test_label = load_org_dat(params)

    return params, trn_dat, test_dat, trn_label, test_label, tidymiss, tidymiss_methods, scaling_methods, imb_methods, Mls_Recommend

def data_info(params,trn_dat,test_dat,trn_label):
    ## 数据基本信息
    missing_ratio = (trn_dat.isnull().sum().sum()) / (len(trn_dat) * len(trn_dat.columns))
    if test_dat is not None:
        num_samples = trn_dat.shape[0] + test_dat.shape[0]
        missing_test_ratio = (test_dat.isnull().sum().sum()) / (len(test_dat) * len(test_dat.columns))
    else:
        num_samples = trn_dat.shape[0]
        missing_test_ratio = 0
    if trn_label is not None:
        ## 判断是否存在标签不平衡
        threshold = 1.5
        label_counts = trn_label['group'].value_counts()
        label_ratios = label_counts.max() / label_counts.min()
        if label_ratios > threshold:
            params['Imbalance'] = True
            if params['Recommend']:
                imb_methods = ['SMOTETomek']
                params['imb_methods'] = imb_methods
        else:
            params['Imbalance'] = False
            params['imb_methods'] = ['None']
    else:
        label_ratios=None

    return params, num_samples, missing_ratio, missing_test_ratio,label_ratios

def data_evaluation(params, trn_dat, test_dat, trn_label, Mls_Recommend):
    ## The basic information of data
    params, num_samples, missing_ratio, missing_test_ratio,label_ratios = data_info(params, trn_dat, test_dat, trn_label)
    ## Recommed MLs
    if params['Recommend']:
        # sample size
        if num_samples < 800:
            sample_size_missing = ['Mean','MI','KNN']
            sample_size_sacler = ['Zscore', 'RobustScaler']
            MLs_list = [method for method in params['ML_Methods'] if method not in {'NeuralNetwork', 'Boruta'}]
            if num_samples <= 600:
                MLs_list = [method for method in MLs_list if method not in {'Logit'}]
            if num_samples <= 300:
                MLs_list = [method for method in MLs_list if method not in {'PLSDA','LASSO'}]
            Mls_Recommend = {
                'Boruta': {
                    'Missing': sample_size_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'GaussianNB': {
                    'Missing': sample_size_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'GBDT': {
                    'Missing': sample_size_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'LASSO': {
                    'Missing': sample_size_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'Logit': {
                    'Missing': sample_size_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'NeuralNetwork': {
                    'Missing': sample_size_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'PLSDA': {
                    'Missing': sample_size_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'RandomForest': {
                    'Missing': sample_size_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'Xgboost': {
                    'Missing': sample_size_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                }
            }
        else:
            MLs_list = params['ML_Methods']

        # imbalance
        if label_ratios > 1.5:
            MLs_list = [method for method in MLs_list if method not in {'GaussianNB', 'NeuralNetwork'}]

        # missing values
        if missing_ratio >= 0.1:
            missing_value_missing = ['MI']
            MLs_list = [method for method in MLs_list if method not in {'GaussianNB', 'NeuralNetwork'}]
            if missing_ratio >= 0.15:
                MLs_list = [method for method in MLs_list if method not in {'LASSO', 'PLSDA'}]
                if missing_ratio >= 0.2:
                    MLs_list = [method for method in MLs_list if method not in {'Logit'}]
                    if missing_ratio >= 0.3:
                        MLs_list = [method for method in MLs_list if method not in {'RandomForest'}]
            Mls_Recommend = {
                'Boruta': {
                    'Missing': missing_value_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'GaussianNB': {
                    'Missing': missing_value_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'GBDT': {
                    'Missing': missing_value_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'LASSO': {
                    'Missing': missing_value_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'Logit': {
                    'Missing': missing_value_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'NeuralNetwork': {
                    'Missing': missing_value_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'PLSDA': {
                    'Missing': missing_value_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'RandomForest': {
                    'Missing': missing_value_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                },
                'Xgboost': {
                    'Missing': missing_value_missing,
                    'Scaling': sample_size_sacler,
                    'imb_methods': ['SMOTETomek']
                }
            }
    else:
        MLs_list = params['ML_Methods']
        params['imb_methods'] = ['SMOTETomek']
        if params['recommendOption'] == 'all':
            tidymiss_methods = ['Mean', 'MI', 'KNN']
            scaling_methods = ['None', 'MinMax', 'Zscore', 'lg', 'RobustScaler', 'MaxAbs', 'PowerTransformer']
            Mls_Recommend = {
                'Boruta': {
                    'Missing': tidymiss_methods,  # 使用 tidymiss_methods
                    'Scaling': scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'GaussianNB': {
                    'Missing': tidymiss_methods,  # 使用 tidymiss_methods
                    'Scaling': scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'GBDT': {
                    'Missing': tidymiss_methods,  # 使用 tidymiss_methods
                    'Scaling': scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'LASSO': {
                    'Missing': tidymiss_methods,  # 使用 tidymiss_methods
                    'Scaling': scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'Logit': {
                    'Missing': tidymiss_methods,  # 使用 tidymiss_methods
                    'Scaling': scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'NeuralNetwork': {
                    'Missing': tidymiss_methods,  # 使用 tidymiss_methods
                    'Scaling': scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'PLSDA': {
                    'Missing': tidymiss_methods,  # 使用 tidymiss_methods
                    'Scaling': scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'RandomForest': {
                    'Missing': tidymiss_methods,  # 使用 tidymiss_methods
                    'Scaling': scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'Xgboost': {
                    'Missing': tidymiss_methods,  # 使用 tidymiss_methods
                    'Scaling': scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                }
            }
        else:
            ex_missing_methods = params['missingValueMethod']
            ex_scaling_methods = params['normalizationMethod']
            Mls_Recommend = {
                'Boruta': {
                    'Missing': ex_missing_methods,  # 使用 tidymiss_methods
                    'Scaling': ex_scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'GaussianNB': {
                    'Missing': ex_missing_methods,  # 使用 tidymiss_methods
                    'Scaling': ex_scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'GBDT': {
                    'Missing': ex_missing_methods,  # 使用 tidymiss_methods
                    'Scaling': ex_scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'LASSO': {
                    'Missing': ex_missing_methods,  # 使用 tidymiss_methods
                    'Scaling': ex_scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'Logit': {
                    'Missing': ex_missing_methods,  # 使用 tidymiss_methods
                    'Scaling': ex_scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'NeuralNetwork': {
                    'Missing': ex_missing_methods,  # 使用 tidymiss_methods
                    'Scaling': ex_scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'PLSDA': {
                    'Missing': ex_missing_methods,  # 使用 tidymiss_methods
                    'Scaling': ex_scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'RandomForest': {
                    'Missing': ex_missing_methods,  # 使用 tidymiss_methods
                    'Scaling': ex_scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                },
                'Xgboost': {
                    'Missing': ex_missing_methods,  # 使用 tidymiss_methods
                    'Scaling': ex_scaling_methods,  # 使用 scaling_methods
                    'imb_methods': ['SMOTETomek']
                }
            }

    return params, MLs_list, missing_ratio, missing_test_ratio,Mls_Recommend,label_ratios