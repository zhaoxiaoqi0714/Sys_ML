import os
from pathlib import Path
import warnings
import pandas as pd
import zipfile
warnings.filterwarnings('ignore')

from myapp.main_processing.data_processing import load_global_params,data_evaluation
from myapp.main_processing.ML_Analysis import ML_data_processing
from myapp.main_processing.SA_Analysis import SA_data_processing
from myapp.pipelines import MLs_pipelines,MLs_res_ana,SAs_pipelines
from myapp.main_processing.Ensemble_Analysis import Ensemble_analysis, Ensemble_res_ana
# project_dir = r'E:\Python_project\Django_Project\data\TCGA-CESC'
def run_analysis(project_dir):
    # Params
    FilePath = project_dir
    Parent_FilePath = Path(FilePath).parent
    params, trn_dat, test_dat, trn_label, test_label, tidymiss, tidymiss_methods, scaling_methods, imb_methods, Mls_Recommend = load_global_params(
        Parent_FilePath, FilePath)
    # data_evaluation
    params, MLs_list, missing_ratio, missing_test_ratio, Mls_Recommend,label_ratios = data_evaluation(params, trn_dat, test_dat,
                                                                                         trn_label, Mls_Recommend)
    print('Missing Ratio: {}'.format(missing_test_ratio))

    # <editor-fold desc="ML analysis">
    if params['MLAnalysisMLAnalysis']:
        SaveFile = os.path.join(params['Parent_FilePath'], params['project_name'],
                                'Results/ML/Plotting/ML_performance_Test.csv')
        if not os.path.isfile(SaveFile):
            print('===================Processing machine analysis===================')
            print('===================Data preparation for Machine learning===================')
            ML_data_processing(params, trn_dat, test_dat, MLs_list, Mls_Recommend, missing_ratio, missing_test_ratio)
            print('Finished data preparation for machine learning.')
            ## processing MLs
            print(
                '===================Processing machine analysis, selected Machine learning methods: {}==================='.format(
                    MLs_list))
            MLs_pipelines(params, MLs_list, Mls_Recommend, missing_ratio, missing_test_ratio, trn_label, test_label)
            print('Finished machine learning in various methods.')
            print('===================Tidying and Plotting ML results===================')
            MLs_res_ana(params, imb_methods)
            print('Finished results analysis of machine learning.')
            if params['Ensemble']:
                print('===================Processing Ensemble Learning===================')
                Single_ML_res = pd.read_csv(SaveFile, index_col=0)
                Ensemble_analysis(params, Single_ML_res, trn_dat, trn_label, test_dat, test_label)
                Ensemble_res_ana(params)
        else:
            if params['Ensemble']:
                print('===================Processing Ensemble Learning===================')
                Single_ML_res = pd.read_csv(SaveFile, index_col=0)
                Ensemble_analysis(params, Single_ML_res, trn_dat, trn_label, test_dat, test_label)
                ## tidying ensemble data
                Ensemble_res_ana(params)
    # </editor-fold>
    # <editor-fold desc="Survival analysis">
    if params['SurvivalAnalysis']:
        if trn_label is not None:
            print('===================Processing Survival analysis===================')
            print('===================Data preparation for Survival analysis===================')
            pre_missings, pre_scalings = SA_data_processing(params, trn_dat, test_dat, missing_ratio,
                                                            missing_test_ratio)
            print('Finished data preparation for Survival analysis.')
            ## processing SAs
            print(

                '===================Processing Survival analysis, selected Survival analysis methods: Multi-Cox regression analysis, Random Forest Survival analysis===================')
            SAs_pipelines(params, missing_ratio, missing_test_ratio, pre_missings, pre_scalings, trn_label, test_label)
            print('Finished Survival Analysis.')
        # </editor-fold>

    # <editor-fold desc="Packing results">
    print('Finished All Analysis Processing!')
    zip_path = pack_folder(FilePath)
    #</editor-fold>

    return zip_path

def pack_folder(FilePath):
    # 生成压缩包路径
    output_zip_path = os.path.join(os.path.dirname(FilePath), str(FilePath.split('\\')[-1]) + '.zip')
    # 创建一个 zip 文件
    with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 遍历文件夹中的所有文件和子文件夹
        for root, dirs, files in os.walk(FilePath):
            for file in files:
                # 获取文件的完整路径
                file_path = os.path.join(root, file)
                # 将文件添加到 zip 文件中，并保留相对路径
                arcname = os.path.relpath(file_path, FilePath)
                zipf.write(file_path, arcname)

    # 返回生成的压缩包路径
    return output_zip_path
