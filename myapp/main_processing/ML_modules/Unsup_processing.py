import os
import umap
import time
import numpy as np
import pandas as pd
from fcmeans import FCM
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder


def Unsup_Nolabel(params, X_trn, ML_scaling, ML_missing, sets):
    results = {}

    # PCA
    start_time = time.time()
    pca_df = Unsup_PCA_Nolabel(params, X_trn, 'group', str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets),
                               use_label=False)
    elapsed = time.time() - start_time
    print(f"PCA completed in {elapsed:.2f} seconds")
    results['PCA'] = pca_df

    # UMAP
    start_time = time.time()
    umap_df = Unsup_UMAP_Nolabel(params, X_trn, 'group', str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets),
                                 use_label=False)
    elapsed = time.time() - start_time
    print(f"UMAP completed in {elapsed:.2f} seconds")
    results['UMAP'] = umap_df

    # NMF
    start_time = time.time()
    nmf_df = Unsup_NMF_Nolabel(params, X_trn, 'group', str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets),
                               use_label=False)
    elapsed = time.time() - start_time
    print(f"NMF completed in {elapsed:.2f} seconds")
    results['NMF'] = nmf_df

    # Kmeans
    start_time = time.time()
    Kmeans_df = Unsup_Kmeans_Nolabel(params, X_trn, 'Cluster',
                                     str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets), use_label=True)
    elapsed = time.time() - start_time
    print(f"Kmeans completed in {elapsed:.2f} seconds")
    results['Kmeans'] = Kmeans_df

    # hierarchy
    start_time = time.time()
    hierarchy_df = Unsup_hierarchy_Nolabel(params, X_trn, 'Cluster',
                                           str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets), use_label=True)
    elapsed = time.time() - start_time
    print(f"Hierarchical clustering completed in {elapsed:.2f} seconds")
    results['hierarchy'] = hierarchy_df

    # DBSCAN
    start_time = time.time()
    DBSCAN_df = Unsup_DBSCAN_Nolabel(params, X_trn, 'Cluster',
                                     str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets), use_label=True)
    elapsed = time.time() - start_time
    print(f"DBSCAN completed in {elapsed:.2f} seconds")
    results['DBSCAN'] = DBSCAN_df

    # FCM
    start_time = time.time()
    FCM_df = Unsup_FCM_Nolabel(params, X_trn, 'Cluster', str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets),
                               use_label=True)
    elapsed = time.time() - start_time
    print(f"FCM completed in {elapsed:.2f} seconds")
    results['FCM'] = FCM_df

    return results


def Unsup_hadTest(params, X_trn, trn_label, ML_scaling, ML_missing, sets):
    results = {}

    # PCA
    start_time = time.time()
    pca_df1 = Unsup_PCA(params, X_trn, trn_label, 'group', str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets),
                        use_label=True)
    pca_df2 = Unsup_PCA(params, X_trn, trn_label, 'group', str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets),
                        use_label=False)
    elapsed = time.time() - start_time
    print(f"PCA (both versions) completed in {elapsed:.2f} seconds")
    results['PCA'] = {'with_label': pca_df1, 'without_label': pca_df2}

    # UMAP
    start_time = time.time()
    umap_df1 = Unsup_UMAP(params, X_trn, trn_label, 'group', str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets),
                          use_label=True)
    umap_df2 = Unsup_UMAP(params, X_trn, trn_label, 'group', str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets),
                          use_label=False)
    elapsed = time.time() - start_time
    print(f"UMAP (both versions) completed in {elapsed:.2f} seconds")
    results['UMAP'] = {'with_label': umap_df1, 'without_label': umap_df2}

    # NMF
    start_time = time.time()
    nmf_df1 = Unsup_NMF(params, X_trn, trn_label, 'group',
                        str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets),
                        use_label=True)
    nmf_df2 = Unsup_NMF(params, X_trn, trn_label, 'group',
                        str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets),
                        use_label=False)
    elapsed = time.time() - start_time
    print(f"NMF (both versions) completed in {elapsed:.2f} seconds")
    results['NMF'] = {'with_label': nmf_df1, 'without_label': nmf_df2}

    # Kmeans
    start_time = time.time()
    Kmeans_df1 = Unsup_Kmeans(params, X_trn, trn_label, 'group',
                              str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets), use_label=True)
    Kmeans_df2 = Unsup_Kmeans(params, X_trn, trn_label, 'Cluster',
                              str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets), use_label=True)
    elapsed = time.time() - start_time
    print(f"Kmeans (both versions) completed in {elapsed:.2f} seconds")
    results['Kmeans'] = {'group': Kmeans_df1, 'Cluster': Kmeans_df2}

    # hierarchy
    start_time = time.time()
    hierarchy_df1 = Unsup_hierarchy(params, X_trn, trn_label, 'group',
                                    str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets), use_label=True)
    hierarchy_df2 = Unsup_hierarchy(params, X_trn, trn_label, 'Cluster',
                                    str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets), use_label=True)
    elapsed = time.time() - start_time
    print(f"Hierarchical clustering (both versions) completed in {elapsed:.2f} seconds")
    results['hierarchy'] = {'group': hierarchy_df1, 'Cluster': hierarchy_df2}

    # DBSCAN
    start_time = time.time()
    DBSCAN_df1 = Unsup_DBSCAN(params, X_trn, trn_label, 'group',
                              str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets), use_label=True)
    DBSCAN_df2 = Unsup_DBSCAN(params, X_trn, trn_label, 'Cluster',
                              str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets), use_label=True)
    elapsed = time.time() - start_time
    print(f"DBSCAN (both versions) completed in {elapsed:.2f} seconds")
    results['DBSCAN'] = {'group': DBSCAN_df1, 'Cluster': DBSCAN_df2}

    # FCM
    start_time = time.time()
    FCM_df1 = Unsup_FCM(params, X_trn, trn_label, 'group', str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets),
                        use_label=True)
    FCM_df2 = Unsup_FCM(params, X_trn, trn_label, 'Cluster', str(ML_missing) + '_' + str(ML_scaling) + '_' + str(sets),
                        use_label=True)
    elapsed = time.time() - start_time
    print(f"FCM (both versions) completed in {elapsed:.2f} seconds")
    results['FCM'] = {'group': FCM_df1, 'Cluster': FCM_df2}

    return results

def Unsup_plotting(params, df, x, y,model, label, mode, use_label = True):
    if params['ML_Plotting']:
        palette = ['#c1121f', '#6a994e', '#5e548e', '#0077b6', '#e76f51']
        if use_label:
            if len(set(df[label])) <= len(palette):
                sel_palette = palette[:len(set(df[label]))]
            else:
                def generate_colors(n, colormap='viridis'):
                    return [plt.cm.get_cmap(colormap)(i / n) for i in range(n)]
                sel_palette = generate_colors(len(set(df[label])))
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df, x=x, y=y, hue=label, palette=sel_palette, legend='full')
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('', fontsize=0)
            plt.xlabel(x, fontsize=16)
            plt.ylabel(y, fontsize=16)
            plt.legend(title='Group', title_fontsize='16', fontsize='13')
            # plt.show()
            plt.savefig(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'{model}-plot-Unsup-results-Group-{mode}.pdf'), dpi=300,
                        bbox_inches='tight')
        else:
            if model == 'PCA':
                df['PC1_PC2_sum'] = df['PC1'] + df['PC2']
            else:
                df['PC1_PC2_sum'] = df['UMAP 1'] + df['UMAP 2']
            sns.set_palette("flare")
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df, x=x, y=y, palette="flare", hue='PC1_PC2_sum', legend=False)
            plt.xticks(fontsize=12)
            plt.yticks(fontsize=12)
            plt.title('', fontsize=0)
            plt.xlabel(x, fontsize=16)
            plt.ylabel(y, fontsize=16)
            # plt.show()
            plt.savefig(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'{model}-plot-Unsup-results-NoGroup-{mode}.pdf'), dpi=300,
                        bbox_inches='tight')

def Unsup_PCA(params, X, label,plt_label, mode, use_label):
    # PCA model
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['group'] = label['group'].tolist()
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup'))
    pca_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'PCA-Unsup results-{mode}.csv'), index=None)

    if params['ML_Plotting']:
        Unsup_plotting(params, pca_df, 'PC1', 'PC2','PCA',plt_label,mode,use_label)

    return pca_df

def Unsup_NMF(params, X, y_true, plt_label, mode, use_label):
    """
    非负矩阵分解 (NMF) 降维，接口与 Unsup_PCA / Unsup_UMAP 完全一致。
    支持:
        use_label=True  -> 用 y_true 作为 group 列
        use_label=False -> 统一用 'NA' 作为 group 列
    自动处理负数（整体平移，使最小值=0）。
    """
    import numpy as np
    from sklearn.decomposition import NMF

    # 1. 处理负数：整体平移到非负
    if np.any(X < 0):
        shift = np.abs(X.min())
        X = X + shift
        print(f"⚠️  Negative values detected; data shifted by +{shift:.4f} to ensure non-negativity.")

    # 2. 拟合 NMF
    nmf_model = NMF(n_components=2, init='nndsvda', random_state=params.get('seed', 42))
    nmf_result = nmf_model.fit_transform(X)          # (n_samples, 2)

    # 3. 组装 DataFrame
    nmf_df = pd.DataFrame(data=nmf_result, columns=["NMF 1", "NMF 2"])
    if use_label and y_true is not None:
        nmf_df["group"] = y_true['group']
    else:
        nmf_df["group"] = 'NA'

    # 4. 保存结果
    out_dir = os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')
    os.makedirs(out_dir, exist_ok=True)
    nmf_df.to_csv(os.path.join(out_dir, f'NMF-Unsup results-{mode}.csv'), index=None)

    # 5. 可选绘图
    if params.get('ML_Plotting', False):
        Unsup_plotting(params, nmf_df, 'NMF 1', 'NMF 2', 'NMF', plt_label, mode, use_label)

    return nmf_df

def Unsup_UMAP(params, X, label,plt_label, mode, use_label):
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    umap_result = umap_model.fit_transform(X)
    umap_df = pd.DataFrame(data=umap_result, columns=["UMAP 1", "UMAP 2"])
    umap_df["group"] = label['group'].tolist()
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup'))
    umap_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'UMAP-Unsup results-{mode}.csv'), index=None)

    if params['ML_Plotting']:
        Unsup_plotting(params, umap_df, 'UMAP 1', 'UMAP 2','UMAP',plt_label,mode,use_label)

    return umap_df

def Unsup_Kmeans(params, X, label, plt_label, mode, use_label):
    # opt cluster numbers
    wcss = []  # 初始化一个列表来存储每个k值对应的WCSS
    for i in range(1, 11):  # 测试1到10个簇
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    wcss_diff = [wcss[i] - wcss[i - 1] for i in range(1, len(wcss))]
    optimal_k_wcss = 1 + wcss_diff.index(max(wcss_diff))
    # kmeans
    kmeans = KMeans(n_clusters=optimal_k_wcss, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_
    # plotting
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    umap_result = umap_model.fit_transform(X)
    umap_df = pd.DataFrame(data=umap_result, columns=["UMAP 1", "UMAP 2"])
    umap_df["group"] = label['group'].tolist()
    umap_df["Cluster"] = labels
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup'))
    umap_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'UMAP-Kmeans results-{mode}.csv'), index=None)
    if params['ML_Plotting']:
        Unsup_plotting(params, umap_df, 'UMAP 1', 'UMAP 2', 'Kmeans', plt_label,str(plt_label)+'-'+mode,use_label)

    return umap_df

def Unsup_hierarchy(params, X, label, plt_label, mode, use_label):
    # opt cluster numbers
    inertia = []
    for i in range(1, 11):  # 测试1到10个簇
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    silhouette_coeffs = []
    for i in range(2, 11):  # 轮廓系数至少需要2个簇
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_coeffs.append(score)
    optimal_k = range(2, 11)[silhouette_coeffs.index(max(silhouette_coeffs))]

    # dendrogram
    dist_matrix = pdist(X, metric='euclidean')
    Z = linkage(dist_matrix, method='ward')
    labels = fcluster(Z, optimal_k, criterion='maxclust')

    # plotting
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    umap_result = umap_model.fit_transform(X)
    umap_df = pd.DataFrame(data=umap_result, columns=["UMAP 1", "UMAP 2"])
    umap_df["group"] = label['group'].tolist()
    umap_df["Cluster"] = labels
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup'))
    umap_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'UMAP-hierarchy results-{mode}.csv'), index=None)
    if params['ML_Plotting']:
        Unsup_plotting(params, umap_df, 'UMAP 1', 'UMAP 2', 'hierarchy', plt_label,str(plt_label)+'-'+mode,use_label)

    return umap_df

def Unsup_DBSCAN(params, X, label, plt_label, mode, use_label):
    # opt cluster numbers
    inertia = []
    for i in range(1, 11):  # 测试1到10个簇
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    silhouette_coeffs = []
    for i in range(2, 11):  # 轮廓系数至少需要2个簇
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_coeffs.append(score)
    optimal_k = range(2, 11)[silhouette_coeffs.index(max(silhouette_coeffs))]

    # dendrogram
    dist_matrix = pdist(X, metric='euclidean')
    Z = linkage(dist_matrix, method='ward')
    labels = fcluster(Z, optimal_k, criterion='maxclust')

    # plotting
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    umap_result = umap_model.fit_transform(X)
    umap_df = pd.DataFrame(data=umap_result, columns=["UMAP 1", "UMAP 2"])
    umap_df["group"] = label['group'].tolist()
    umap_df["Cluster"] = labels
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup'))
    umap_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'UMAP-DBSCAN results-{mode}.csv'), index=None)
    if params['ML_Plotting']:
        Unsup_plotting(params, umap_df, 'UMAP 1', 'UMAP 2', 'DBSCAN', plt_label,str(plt_label)+'-'+mode,use_label)

    return umap_df

def Unsup_FCM(params, X, label, plt_label, mode, use_label):
    fcm = FCM(n_clusters=3)
    fcm.fit(X)
    centers = fcm.centers
    u = fcm.u  # 隶属度矩阵
    fpc = np.sum(u ** 2) / (X.shape[0] * u.shape[0])
    labels = np.argmax(u, axis=1)
    # plotting
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    umap_result = umap_model.fit_transform(X)
    umap_df = pd.DataFrame(data=umap_result, columns=["UMAP 1", "UMAP 2"])
    umap_df["group"] = label['group'].tolist()
    umap_df["Cluster"] = labels
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup'))
    umap_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'UMAP-FCM results-{mode}.csv'), index=None)
    if params['ML_Plotting']:
        Unsup_plotting(params, umap_df, 'UMAP 1', 'UMAP 2', 'FCM', plt_label,str(plt_label)+'-'+mode,use_label)

    return umap_df

def Unsup_PCA_Nolabel(params, X,plt_label, mode, use_label):
    # PCA model
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    pca_df['group'] = 'NA'
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup'))
    pca_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'PCA-Unsup results-{mode}.csv'), index=None)

    if params['ML_Plotting']:
        Unsup_plotting(params, pca_df, 'PC1', 'PC2','PCA',plt_label,mode,use_label)

    return pca_df

def Unsup_UMAP_Nolabel(params, X, plt_label, mode, use_label):
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    umap_result = umap_model.fit_transform(X)
    umap_df = pd.DataFrame(data=umap_result, columns=["UMAP 1", "UMAP 2"])
    umap_df["group"] = 'NA'
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup'))
    umap_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'UMAP-Unsup results-{mode}.csv'), index=None)
    if params['ML_Plotting']:
        Unsup_plotting(params, umap_df, 'UMAP 1', 'UMAP 2','UMAP',plt_label,mode,use_label)

    return umap_df

def Unsup_Kmeans_Nolabel(params, X,  plt_label, mode, use_label):
    # opt cluster numbers
    wcss = []  # 初始化一个列表来存储每个k值对应的WCSS
    for i in range(1, 11):  # 测试1到10个簇
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)
    wcss_diff = [wcss[i] - wcss[i - 1] for i in range(1, len(wcss))]
    optimal_k_wcss = 1 + wcss_diff.index(max(wcss_diff))
    # kmeans
    kmeans = KMeans(n_clusters=optimal_k_wcss, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    labels = kmeans.labels_
    # plotting
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    umap_result = umap_model.fit_transform(X)
    umap_df = pd.DataFrame(data=umap_result, columns=["UMAP 1", "UMAP 2"])
    umap_df["group"] =  'NA'
    umap_df["Cluster"] = labels
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup'))
    umap_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'UMAP-Kmeans results-{mode}.csv'), index=None)
    if params['ML_Plotting']:
        Unsup_plotting(params, umap_df, 'UMAP 1', 'UMAP 2', 'Kmeans', plt_label,str(plt_label)+'-'+mode,use_label)

    return umap_df

def Unsup_hierarchy_Nolabel(params, X,plt_label, mode, use_label):
    # opt cluster numbers
    inertia = []
    for i in range(1, 11):  # 测试1到10个簇
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    silhouette_coeffs = []
    for i in range(2, 11):  # 轮廓系数至少需要2个簇
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_coeffs.append(score)
    optimal_k = range(2, 11)[silhouette_coeffs.index(max(silhouette_coeffs))]

    # dendrogram
    dist_matrix = pdist(X, metric='euclidean')
    Z = linkage(dist_matrix, method='ward')
    labels = fcluster(Z, optimal_k, criterion='maxclust')

    # plotting
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    umap_result = umap_model.fit_transform(X)
    umap_df = pd.DataFrame(data=umap_result, columns=["UMAP 1", "UMAP 2"])
    umap_df["group"] =  'NA'
    umap_df["Cluster"] = labels
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup'))
    umap_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'UMAP-hierarchy results-{mode}.csv'), index=None)
    if params['ML_Plotting']:
        Unsup_plotting(params, umap_df, 'UMAP 1', 'UMAP 2', 'hierarchy', plt_label,str(plt_label)+'-'+mode,use_label)

    return umap_df

def Unsup_DBSCAN_Nolabel(params, X, plt_label, mode, use_label):
    # opt cluster numbers
    inertia = []
    for i in range(1, 11):  # 测试1到10个簇
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
    silhouette_coeffs = []
    for i in range(2, 11):  # 轮廓系数至少需要2个簇
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_coeffs.append(score)
    optimal_k = range(2, 11)[silhouette_coeffs.index(max(silhouette_coeffs))]

    # dendrogram
    dist_matrix = pdist(X, metric='euclidean')
    Z = linkage(dist_matrix, method='ward')
    labels = fcluster(Z, optimal_k, criterion='maxclust')

    # plotting
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    umap_result = umap_model.fit_transform(X)
    umap_df = pd.DataFrame(data=umap_result, columns=["UMAP 1", "UMAP 2"])
    umap_df["group"] = 'NA'
    umap_df["Cluster"] = labels
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup'))
    umap_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'UMAP-DBSCAN results-{mode}.csv'), index=None)
    if params['ML_Plotting']:
        Unsup_plotting(params, umap_df, 'UMAP 1', 'UMAP 2', 'DBSCAN', plt_label,str(plt_label)+'-'+mode,use_label)

    return umap_df

def Unsup_FCM_Nolabel(params, X, plt_label, mode, use_label):
    fcm = FCM(n_clusters=3)
    fcm.fit(X)
    centers = fcm.centers
    u = fcm.u  # 隶属度矩阵
    fpc = np.sum(u ** 2) / (X.shape[0] * u.shape[0])
    labels = np.argmax(u, axis=1)
    # plotting
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='euclidean')
    umap_result = umap_model.fit_transform(X)
    umap_df = pd.DataFrame(data=umap_result, columns=["UMAP 1", "UMAP 2"])
    umap_df["group"] =  'NA'
    umap_df["Cluster"] = labels
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup'))
    umap_df.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup', f'UMAP-FCM results-{mode}.csv'), index=None)
    if params['ML_Plotting']:
        Unsup_plotting(params, umap_df, 'UMAP 1', 'UMAP 2', 'FCM', plt_label,str(plt_label)+'-'+mode,use_label)

    return umap_df

def Unsup_NMF_Nolabel(params, X, plt_label, mode, use_label):
    """
    无监督 NMF 降维（2 维），保存结果，可选绘图。
    接口与 Unsup_UMAP_Nolabel 保持一致。
    """
    # 1. 基础检查：NMF 要求非负输入
    if (X < 0).any():
        shift = np.abs(X.min())
        X = X + shift
        print(f"⚠️  Negative values detected; data shifted by +{shift:.4f} to ensure non-negativity.")

    # 2. 拟合 NMF
    nmf_model = NMF(n_components=2, init='nndsvda', random_state=params.get('seed', 42))
    nmf_result = nmf_model.fit_transform(X)          # shape: (n_samples, 2)

    # 3. 包装成 DataFrame，方便后续操作
    nmf_df = pd.DataFrame(data=nmf_result, columns=["NMF 1", "NMF 2"])
    nmf_df["group"] = 'NA'                           # 与 UMAP 保持一致

    # 4. 存储结果
    out_dir = os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/Unsup')
    os.makedirs(out_dir, exist_ok=True)
    nmf_df.to_csv(os.path.join(out_dir, f'NMF-Unsup results-{mode}.csv'), index=None)

    # 5. 可选绘图（复用原 Unsup_plotting 函数）
    if params.get('ML_Plotting', False):
        Unsup_plotting(params, nmf_df, 'NMF 1', 'NMF 2', 'NMF', plt_label, mode, use_label)

    return nmf_df
