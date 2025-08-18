import os
import glob
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,LabelEncoder
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_recall_curve, auc, roc_curve
from scipy.stats import pearsonr
import pyecharts.options as opts
from pyecharts.charts import Bar3D, Tab, HeatMap,Grid

def load_ML_res(params):
    csv_files = glob.glob(os.path.join(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML'), '*.csv'))
    csv_files = [os.path.basename(file_path) for file_path in csv_files]
    features_scores_files = []
    test_data_files = []
    train_data_files = []
    for file_path in csv_files:
        if file_path.startswith('Features Scores'):
            features_scores_files.append(file_path)
        elif file_path.startswith('Test dat'):
            test_data_files.append(file_path)
        elif file_path.startswith('Trn dat'):
            train_data_files.append(file_path)
    return features_scores_files,test_data_files,train_data_files

def tidying_ML_res(params, features_scores_files, test_data_files, train_data_files):
    # tidying features_scores
    final_df = feat_scores_df(params,features_scores_files)
    # tidying trn_dat
    trn_df = samples_res(params,train_data_files)
    # tidying tst_dat
    tst_df = samples_res(params,test_data_files)

    return final_df,trn_df,tst_df

def tidying_feats_res(params,final_df):
    pivot_df = final_df.pivot_table(index='Feat', columns='Condition', values='Scores')
    pivot_df = pivot_df.replace([np.inf, -np.inf], np.nan)
    pivot_df = pivot_df.fillna(0)
    correlation_matrix = pivot_df.corr()
    results = []
    for i in range(len(pivot_df.columns) - 1):
        for j in range(i + 1, len(pivot_df.columns)):
            corr_coef, p_value = pearsonr(pivot_df.iloc[:, i], pivot_df.iloc[:, j])
            Condition1 = pivot_df.columns[i]
            Condition2 = pivot_df.columns[j]
            results.append([Condition1, Condition2, corr_coef, p_value])
    results_df = pd.DataFrame(results, columns=['Condition1', 'Condition2', 'Correlation', 'P-value'])
    results_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML/Plotting/ML_Feats_corr.csv'), index=None)

    return results_df

def feat_scores_df(params,features_scores_files):
    final_df = pd.DataFrame()
    for fs_file in features_scores_files:
        # obtained info
        parts = [part.strip() for part in fs_file.split("-")]
        model = parts[1]
        mode = parts[3].split('.csv')[0]
        Condition = str(model) + '_' + str(mode)

        # tidying df
        scaler = MinMaxScaler()
        fs_df = pd.read_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML', fs_file))
        fs_df = fs_df[['Feat', 'Scores']]
        fs_df['Scores'] = fs_df['Scores'].abs()
        fs_df['Scores'] = scaler.fit_transform(fs_df[['Scores']].values)
        fs_df['Condition'] = Condition
        # 纵向拼接DataFrame
        final_df = pd.concat([final_df, fs_df], ignore_index=True)
    return final_df

def samples_res(params,data_files):
    trn_df = pd.DataFrame()
    for trn_file in data_files:
        # obtained info
        parts = [part.strip() for part in trn_file.split("-")]
        model = parts[1]
        mode = parts[3].split('.csv')[0]
        Condition = str(model) + '_' + str(mode)
        # tidying df
        fs_df = pd.read_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML', trn_file))
        if params['HadTest']:
            fs_df.columns = ['True_label', 'Pred_label','Prob']
        else:
            fs_df.columns = ['Fold','Prob','Pred_label','True_label']
        fs_df['Condition'] = Condition
        # 纵向拼接DataFrame
        trn_df = pd.concat([trn_df, fs_df], ignore_index=True)

    return trn_df

def ml_matrics(trn_df):
    class_num = len(set(trn_df['True_label']))
    if class_num < 3:
        trn_df['True_label'] = trn_df['True_label'].astype('category')
        trn_df['Pred_label'] = trn_df['Pred_label'].astype('category')
        le = LabelEncoder()
        trn_df['True_label_encoded'] = le.fit_transform(trn_df['True_label'])
        trn_df['Pred_label_encoded'] = le.transform(trn_df['Pred_label'])
        metrics_dict = {
            'AUC': [],
            'F1_score': [],
            'PR-AUC': [],
            'Accuracy': []
        }
        # 对每个 Condition 进行迭代
        for condition in trn_df['Condition'].unique():
            # 筛选当前 Condition 的数据
            condition_df = trn_df[trn_df['Condition'] == condition]

            # 计算 AUC
            if len(condition_df['True_label_encoded'].unique()) < 2:
                auc_score = np.nan
            else:
                auc_score = roc_auc_score(condition_df['True_label_encoded'], condition_df['Prob'])
            metrics_dict['AUC'].append(auc_score)

            # 计算 F1 分数
            f1 = f1_score(condition_df['True_label_encoded'], condition_df['Pred_label_encoded'])
            metrics_dict['F1_score'].append(f1)

            # 计算准确率
            accuracy = accuracy_score(condition_df['True_label_encoded'], condition_df['Pred_label_encoded'])
            metrics_dict['Accuracy'].append(accuracy)

            # 计算 PR-AUC
            precision, recall, _ = precision_recall_curve(condition_df['True_label_encoded'],
                                                          condition_df['Prob'])
            pr_auc = auc(recall, precision)
            metrics_dict['PR-AUC'].append(pr_auc)

        # 将结果转换为 DataFrame
        metrics_df = pd.DataFrame(metrics_dict, index=trn_df['Condition'].unique())
    else:
        metrics_dict = {
            'AUC': [],
            'F1_score': [],
            'PR-AUC': [],
            'Accuracy': []
        }
        # 对每个 Condition 进行迭代
        for condition in trn_df['Condition'].unique():
            # 筛选当前 Condition 的数据
            condition_df = trn_df[trn_df['Condition'] == condition]
            is_text = list(set(condition_df['Pred_label'].apply(lambda x: isinstance(x, str))))
            if is_text[0]:
                label_encoder = LabelEncoder()
                all_labels = pd.concat([condition_df['Pred_label'], condition_df['True_label']]).unique()
                label_encoder.fit(all_labels)
                condition_df['Pred_label_encoded'] = label_encoder.transform(condition_df['Pred_label'])
                condition_df['True_label_encoded'] = label_encoder.transform(condition_df['True_label'])
                condition_df['Prob_array'] = condition_df['Prob'].apply(parse_prob)
                y_prob = np.vstack(condition_df['Prob_array'].values)
                auc_score = roc_auc_score(
                    condition_df['True_label_encoded'],
                    y_prob,
                    multi_class='ovo'  # 或 'ovr'
                    )
                metrics_dict['AUC'].append(auc_score)
                f1 = f1_score(condition_df['True_label_encoded'], condition_df['Pred_label_encoded'], average='micro')
                metrics_dict['F1_score'].append(f1)
                accuracy = accuracy_score(condition_df['True_label_encoded'], condition_df['Pred_label_encoded'])
                metrics_dict['Accuracy'].append(accuracy)
                metrics_dict['PR-AUC'].append('NaN')
            else:
                metrics_dict['PR-AUC'].append('NaN')
                y_prob = np.column_stack([
                    1 - np.clip(condition_df['Pred_label'], 0, 1),  # 类0概率
                    np.where((condition_df['Pred_label'] > 0.5) & (condition_df['Pred_label'] <= 1.5),
                             condition_df['Pred_label'], 0),  # 类1概率
                    np.clip(condition_df['Pred_label'] - 1.5, 0, None)  # 类2概率
                ])
                y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)

                y_true = condition_df['True_label'].values
                n_classes = len(np.unique(y_true))
                optimal_thresholds = []
                youden_indices = []
                for i in range(n_classes):
                    # 二值化当前类别
                    y_true_bin = (y_true == i).astype(int)
                    fpr, tpr, thresholds = roc_curve(y_true_bin, y_prob[:, i])

                    # 计算Youden's Index (J = TPR - FPR)
                    j_scores = tpr - fpr
                    optimal_idx = np.argmax(j_scores)
                    optimal_thresholds.append(thresholds[optimal_idx])
                    youden_indices.append(j_scores[optimal_idx])

                    print(
                        f"Class {i}: Optimal threshold = {thresholds[optimal_idx]:.4f}, Youden's Index = {j_scores[optimal_idx]:.4f}")
                # 根据最佳阈值预测类别
                y_pred = np.zeros_like(y_true)
                for i in range(n_classes):
                    y_pred[y_prob[:, i] >= optimal_thresholds[i]] = i

                # 处理可能的多重赋值（取概率最高的类别）
                conflict_mask = (y_prob >= optimal_thresholds).sum(axis=1) > 1
                y_pred[conflict_mask] = np.argmax(y_prob[conflict_mask], axis=1)
                accuracy = accuracy_score(np.array(y_true, dtype=int), np.array(y_pred, dtype=int))
                f1_macro = f1_score(np.array(y_true, dtype=int), np.array(y_pred, dtype=int), average='macro')
                auc_score = roc_auc_score(
                    np.array(y_true, dtype=int),
                    y_prob,
                    multi_class='ovo'  # 或 'ovr'
                    )
                metrics_dict['AUC'].append(auc_score)
                metrics_dict['F1_score'].append(f1_macro)
                metrics_dict['Accuracy'].append(accuracy)

        # 将结果转换为 DataFrame
        metrics_df = pd.DataFrame(metrics_dict, index=trn_df['Condition'].unique())

    return metrics_df,trn_df

def parse_prob(x):
    try:
        # 移除多余字符并分割
        clean = x.replace('[', '').replace(']', '').strip()
        numbers = [float(num) for num in clean.split() if num]
        return np.array(numbers)
    except:
        print(f"Failed to parse: {x}")
        return np.array([0., 0., 0.])  # 返回默认值

def tidying_ml_plotting_df(trn_metrics_df):
    trn_metrics_df_reset = trn_metrics_df.reset_index()
    trn_metrics_df_reset[['Model', 'MissingMethod', 'ScalingMethod', 'ImbalanceMethod']] = trn_metrics_df_reset[
        'index'].str.split('_', expand=True)
    trn_metrics_df_reset = trn_metrics_df_reset.drop(columns=['index'])
    trn_metrics_df_reset.loc[
        trn_metrics_df_reset["ImbalanceMethod"].isin(["Borderline", "Borderline-SMOTE"]),
        "ImbalanceMethod"
    ] = "Borderline-SMOTE"
    trn_metrics_df_reset.loc[
        trn_metrics_df_reset["ImbalanceMethod"].isin(["SMOTEENN", "SMOTE"]),
        "ImbalanceMethod"
    ] = "SMOTE"
    # tidying dat
    le_missing = LabelEncoder()
    le_scaling = LabelEncoder()
    trn_metrics_df_reset['MissingMethod_encoded'] = le_missing.fit_transform(trn_metrics_df_reset['MissingMethod'])
    trn_metrics_df_reset['ScalingMethod_encoded'] = le_scaling.fit_transform(trn_metrics_df_reset['ScalingMethod'])

    return trn_metrics_df_reset,le_missing,le_scaling

def Sys_ML_matrics_plotting(params, trn_metrics_df,imb_methods, sets):
    trn_metrics_df_reset, le_missing, le_scaling = tidying_ml_plotting_df(trn_metrics_df)

    ml_plotting(params, 'AUC', trn_metrics_df_reset, le_scaling, le_missing, imb_methods, sets)
    ml_plotting(params, 'F1_score', trn_metrics_df_reset, le_scaling, le_missing, imb_methods, sets)
    ml_plotting(params, 'PR-AUC', trn_metrics_df_reset, le_scaling, le_missing, imb_methods, sets)
    ml_plotting(params, 'Accuracy', trn_metrics_df_reset, le_scaling, le_missing, imb_methods, sets)

def Sys_ML_feats_ploting(params,results_df):
    same_methods_df = diff_methods_df = None
    if results_df.shape[0]>0:
        # splitted dat
        same_methods_df, diff_methods_df = Sys_ML_feats_pre_dat(results_df)
        # plotting same dat
        x_methods, y_methods, values = same_plt_dat(same_methods_df)
        feat_heat_plt(params, x_methods, y_methods, values, 'Same_models')
        # plotting diff methods
        x_methods, y_methods, values = diff_plt_dat(diff_methods_df)
        feat_heat_plt(params, x_methods, y_methods, values, 'diff_models')

    return same_methods_df, diff_methods_df

def ml_plotting(params, indicator,trn_metrics_df_reset,le_scaling,le_missing,imb_methods,sets):
    auc_df = trn_metrics_df_reset[['ScalingMethod_encoded', 'MissingMethod_encoded', indicator, 'Model', 'ImbalanceMethod']]
    ScalingMethod = list(le_scaling.inverse_transform(list(set(trn_metrics_df_reset['ScalingMethod_encoded']))))
    MissingMethod = list(le_missing.inverse_transform(list(set(trn_metrics_df_reset['MissingMethod_encoded']))))
    ImbalanceMethod = list(set(auc_df['ImbalanceMethod']))

    # prepared plotting set
    imb_colors = ['#e9edc9', '#f4a259', '#219ebc', '#a5668b', '#8cb369', '#f07167', '#ccdb33', '#007f5f', '#eac435']
    imb_colors_dict = dict(zip(imb_methods, imb_colors))

    # prepared dat
    z_dat_dict = {}
    for index, row in auc_df.iterrows():
        model = row['Model']
        # 检查当前 Model 类型是否已经在字典中，如果没有则添加
        if model not in z_dat_dict:
            z_dat_dict[model] = []
        # 将当前行的数据添加到对应 Model 类型的 z_dat 列表中
        if len(ImbalanceMethod) > 1:
            z_dat_dict[model].append([
                row['ScalingMethod_encoded'],
                row['MissingMethod_encoded'],
                row[indicator],
                row['ImbalanceMethod']
            ])
        else:
            z_dat_dict[model].append([
                row['ScalingMethod_encoded'],
                row['MissingMethod_encoded'],
                row[indicator]
            ])

    # plotting
    tab = Tab()
    for model, z_dat in z_dat_dict.items():
        tab.add(bar3D_slider(z_dat,ImbalanceMethod, ScalingMethod,MissingMethod,imb_methods,indicator,imb_colors_dict), model)
    tab.render(os.path.join(params['Parent_FilePath'], params['project_name'], f"Results/ML/Plotting/ML_Plotting_{indicator}_{sets}.html"))

def bar3D_slider(z_dat,ImbalanceMethod, ScalingMethod,MissingMethod,imb_methods,indicator,imb_colors_dict):
    '''
    目前关于Imb的颜色配置可能有问题
    '''
    if len(ImbalanceMethod) > 1:
        # extracted plotting dat
        c = Bar3D()
        colors = []
        for imb in ImbalanceMethod:
            filtered_data = [item for item in z_dat if item[3] == imb]
            ex_dat = [item[:3] for item in filtered_data]
            c.add(
                series_name=imb,
                data=ex_dat,
                shading="lambert",
                xaxis3d_opts=opts.Axis3DOpts(data=ScalingMethod, type_="category", name="Scaling Method"),
                yaxis3d_opts=opts.Axis3DOpts(data=MissingMethod, type_="category", name="Missing Method"),
                zaxis3d_opts=opts.Axis3DOpts(type_="value", name=indicator, min_=0.6, max_=1 * len(ImbalanceMethod)),
            ).set_series_opts(
                **{"stack": "stack"}
            )
            colors.append(imb_colors_dict[imb])
        c.set_colors(colors)
        c.set_global_opts(
                title_opts=opts.TitleOpts(title=""),
                legend_opts=opts.LegendOpts(is_show=True)  # 显示图例
            )
            # .render(os.path.join(r'E:\Python_project\Systematic_ML\test',"bar3d_punch_card.html"))
    else:
        c = (
            Bar3D()
            .add(
                series_name="",
                data=z_dat,
                xaxis3d_opts=opts.Axis3DOpts(type_="category", data=ScalingMethod, name="Scaling Method"),
                yaxis3d_opts=opts.Axis3DOpts(type_="category", data=MissingMethod, name="Missing Method"),
                zaxis3d_opts=opts.Axis3DOpts(type_="value", name=indicator, min_=0.6, max_=1),
            )
            .set_global_opts(
                visualmap_opts=opts.VisualMapOpts(
                    max_=1,
                    range_color=[
                        "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
                        "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"
                    ],
                )
            )
            # .render(os.path.join(r'E:\Python_project\Systematic_ML\data\T2D\test',"bar3d_punch_card.html"))
        )
    return c

def Sys_ML_feats_pre_dat(results_df):
    same_methods_df = diff_methods_df = None

    if results_df.shape[0] > 0:
        results_df[['C1_Model', 'C1_MissingMethod', 'C1_ScalingMethod', 'C1_ImbalanceMethod']] = results_df[
            'Condition1'].str.split('_', expand=True)
        results_df[['C2_Model', 'C2_MissingMethod', 'C2_ScalingMethod', 'C2_ImbalanceMethod']] = results_df[
            'Condition2'].str.split('_', expand=True)
        results_df = results_df[results_df['P-value'] < 0.05]

        same_methods_df = results_df[results_df['C1_Model'] == results_df['C2_Model']]
        diff_methods_df = results_df[results_df['C1_Model'] != results_df['C2_Model']]

    return same_methods_df,diff_methods_df

def same_plt_dat(same_methods_df):
    # plotting same methods
    Model_list = list(set(same_methods_df['C1_Model']))
    same_methods_df['X_label'] = same_methods_df['C1_MissingMethod'].astype(str) + "_" + \
                                 same_methods_df['C1_ScalingMethod'].astype(str) + "_" + \
                                 same_methods_df['C1_ImbalanceMethod'].astype(str)
    same_methods_df['Y_label'] = same_methods_df['C2_MissingMethod'].astype(str) + "_" + \
                                 same_methods_df['C2_ScalingMethod'].astype(str) + "_" + \
                                 same_methods_df['C2_ImbalanceMethod'].astype(str)

    le = LabelEncoder()
    all_labels = list(set(same_methods_df['X_label']) | set(same_methods_df['Y_label']))
    le.fit(all_labels)
    same_methods_df['X_label_encoded'] = le.transform(same_methods_df['X_label'])
    same_methods_df['Y_label_encoded'] = le.transform(same_methods_df['Y_label'])
    x_methods = list(le.inverse_transform(list(set(same_methods_df['X_label_encoded']))))
    y_methods = list(le.inverse_transform(list(set(same_methods_df['Y_label_encoded']))))
    values = {}
    # 根据 Model_list 提取数据并添加到 values
    for model in Model_list:
        model_data = same_methods_df[same_methods_df['C1_Model'] == model]
        value = []
        for index, row in model_data.iterrows():
            value.append([row['X_label_encoded'], row['Y_label_encoded'], round(row['Correlation'], 3)])
        values.update({model: value})
    return x_methods, y_methods, values

def diff_plt_dat(diff_methods_df):
    diff_methods_df['Prepared_Method1'] = diff_methods_df['C1_MissingMethod'].astype(str) + "_" + \
                                          diff_methods_df['C1_ScalingMethod'].astype(str) + "_" + \
                                          diff_methods_df['C1_ImbalanceMethod'].astype(str)
    diff_methods_df['Prepared_Method2'] = diff_methods_df['C2_MissingMethod'].astype(str) + "_" + \
                                          diff_methods_df['C2_ScalingMethod'].astype(str) + "_" + \
                                          diff_methods_df['C2_ImbalanceMethod'].astype(str)
    all_labels = pd.concat([diff_methods_df['C1_Model'], diff_methods_df['C2_Model']], axis=0).unique()
    le = LabelEncoder()
    le.fit(all_labels)
    diff_methods_df['X_label_encoded'] = le.transform(diff_methods_df['C1_Model'])
    diff_methods_df['Y_label_encoded'] = le.transform(diff_methods_df['C2_Model'])
    x_methods = list(le.inverse_transform(list(set(diff_methods_df['X_label_encoded']))))
    y_methods = list(le.inverse_transform(list(set(diff_methods_df['Y_label_encoded']))))
    values = {}
    Prepared_Method_list = list(set(diff_methods_df['Prepared_Method1']))
    # 根据 Model_list 提取数据并添加到 values
    for model in Prepared_Method_list:
        model_data = diff_methods_df[
            (diff_methods_df['Prepared_Method1'] == model) & (diff_methods_df['Prepared_Method2'] == model)]
        value = []
        for index, row in model_data.iterrows():
            value.append([row['X_label_encoded'], row['Y_label_encoded'], round(row['Correlation'], 3)])
        values.update({model: value})
    return x_methods, y_methods, values

def feat_heat_plt(params, x_methods, y_methods, values, sets):
    c = HeatMap(
        init_opts=opts.InitOpts(width="1300px", height="800px")
    )
    # 遍历 values 字典的键，为每个模型添加数据
    for model, dat in values.items():
        c.add_xaxis(x_methods)  # 添加 x 轴数据
        c.add_yaxis(
            model,
            y_methods,
            dat,
            label_opts=opts.LabelOpts(is_show=True, position="inside"),
        )
    if len(values) < 10:
        pos_top = "6%"
        pos_left = "32%"
    else:
        pos_top = "16%"
        pos_left = "10%"

    c.set_global_opts(
        title_opts=opts.TitleOpts(title=""),
        visualmap_opts=opts.VisualMapOpts(
            min_=-1, max_=1, is_calculable=True, orient="vertical", pos_right="10%", pos_top='40%',
            item_width=20, item_height=200,
            range_color=[
                "#313695", "#4575b4", "#74add1", "#abd9e9", "#e0f3f8",
                "#ffffbf", "#fee090", "#fdae61", "#f46d43", "#d73027", "#a50026"
            ]
        ),
        xaxis_opts=opts.AxisOpts(
            type_="category",
            splitarea_opts=opts.SplitAreaOpts(
                is_show=True, areastyle_opts=opts.AreaStyleOpts(opacity=0)
            ),
            axislabel_opts=opts.LabelOpts(interval=0, font_size=12, rotate=45),
        ),
        legend_opts=opts.LegendOpts(orient="horizontal", pos_top="2%", pos_left=pos_left, pos_bottom="100px")
    )
    # c.render(os.path.join(args.project, 'Results/ML/Plotting/test.html'))
    grid = (
        Grid(init_opts=opts.InitOpts(width="1300px", height="800px"))
    ).add(
        c, grid_opts=opts.GridOpts(pos_top=pos_top, pos_bottom="150px", pos_left="200px", pos_right="200px")
    )
    grid.render(os.path.join(params['Parent_FilePath'], params['project_name'], f'Results/ML/Plotting/ML_Feats_corr_heatmap_{sets}.html'))