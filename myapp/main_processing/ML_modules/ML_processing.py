import os
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb
from sklearn.svm import SVC
from boruta import BorutaPy
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import roc_curve

from myapp.main_processing.ML_modules.metrics import cal_youden, ROC, extract_imp_feat,compute_VIP

def save_res(params,feat_res,trn_res,tst_res,y_trn, y_prob_trn,test_label,y_prob_tst,mode,model):
    # save results
    feat_res.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML',
                                 f'Features Scores-{model}-ML results-{mode}.csv'),
                    index=None)
    trn_res.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML',
                                f'Trn dat-{model}-ML results-{mode}.csv'),
                   index=None)
    tst_res.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML',
                                f'Test dat-{model}-ML results-{mode}.csv'),
                   index=None)
    # plotting
    if params['ML_Plotting']:
        roc_res_trn = ROC(y_trn, y_prob_trn, params, model, mode, 'Train')
        roc_res_tst = ROC(test_label['group_encoded'].values, y_prob_tst, params,
                          model, mode, 'Test')
        roc_res = {'Train': roc_res_trn, 'Test': roc_res_tst}

def Boruta_ML(params,trn_dat, X_trn, y_trn,X_tst,test_label,y_tst,le,mode):
    feat_res = trn_res = tst_res = roc_res = None
    # constructed model
    param_grid = {
        'n_estimators': [100, 200, 300],  # 决策树的数量
        'max_depth': [None, 5, 10, 20],  # 树的最大深度
        'min_samples_split': [2, 5, 10],  # 分割内部节点所需的最小样本数
        'min_samples_leaf': [1, 2, 4],  # 叶子节点所需的最小样本数
        'max_features': ['auto', 'sqrt', 'log2']  # 寻找最佳分割时要考虑的特征数量
    }
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5,
                               scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_trn, y_trn)
    best_rf = grid_search.best_estimator_
    boruta_selector = BorutaPy(best_rf, n_estimators='auto', verbose=0,
                               random_state=1, max_iter=50)
    boruta_selector.fit(X_trn, y_trn)
    selected_features = boruta_selector.support_
    feature_ranks = boruta_selector.ranking_
    if np.sum(selected_features) > 0:
        X_train_selected = boruta_selector.transform(X_trn)
        X_test_selected = boruta_selector.transform(X_tst)
    else:
        X_train_selected = X_trn
        X_test_selected = X_tst

    rf.fit(X_train_selected, y_trn)
    if y_trn.max() + 1 < 3:
        y_prob_trn = rf.predict_proba(X_train_selected)[:, 1]
        y_prob_tst = rf.predict_proba(X_test_selected)[:, 1]
        y_pred_trn, y_pred_tst = cal_pred(y_trn, y_prob_trn, y_prob_tst, y_tst)
    else:
        y_prob_trn = rf.predict_proba(X_train_selected)
        y_prob_tst = rf.predict_proba(X_test_selected)
        y_pred_trn = np.argmax(y_prob_trn, axis=1)
        y_pred_tst = np.argmax(y_prob_tst, axis=1)

    ## save results
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML'))
    # features
    feat_res = pd.DataFrame({
        'Feat': trn_dat.columns,
        'Scores': feature_ranks
    })
    # predicted_values
    trn_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_trn),
        'Pred_label': le.inverse_transform(y_pred_trn),
        'Prob': y_prob_trn if (y_trn.max() + 1) < 3 else y_pred_trn
    })
    tst_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_tst),
        'Pred_label': le.inverse_transform(y_pred_tst),
        'Prob': y_prob_tst if (y_trn.max() + 1) < 3 else y_pred_tst
    })
    # save results
    save_res(params,feat_res,trn_res,tst_res,y_trn, y_prob_trn,test_label,y_prob_tst,mode,'Boruta')

    return feat_res, trn_res, tst_res, roc_res

def GaussianNB_ML(params,trn_dat, X_trn, y_trn,X_tst,test_label,y_tst,le,mode):
    feat_res = trn_res = tst_res = roc_res = None
    # constructed model
    param_grid = {
        'priors': [None, [0.25, 0.75], [0.75, 0.25]]
    }
    nb = GaussianNB()
    grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_trn, y_trn)
    best_nb = grid_search.best_estimator_
    if y_trn.max() +1 < 3:
        y_prob_trn = best_nb.predict_proba(X_trn)[:, 1]
        y_prob_tst = best_nb.predict_proba(X_tst)[:, 1]
        y_pred_trn, y_pred_tst = cal_pred(y_trn,y_prob_trn,y_prob_tst,y_tst)
    else:
        y_prob_trn = best_nb.predict_proba(X_trn)
        y_prob_tst = best_nb.predict_proba(X_tst)
        y_pred_trn = np.argmax(y_prob_trn, axis=1)
        y_pred_tst = np.argmax(y_prob_tst, axis=1)

    ## save results
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML'))
    # features
    feat_res = pd.DataFrame({
        'Feat': trn_dat.columns,
        'Scores': 'NA'
    })
    # predicted_values
    trn_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_trn),
        'Pred_label': le.inverse_transform(y_pred_trn),
        'Prob': y_prob_trn if (y_trn.max()+1) < 3 else y_pred_trn
    })
    tst_res = pd.DataFrame({
        'True_label':le.inverse_transform(y_tst),
        'Pred_label': le.inverse_transform(y_pred_tst),
        'Prob': y_prob_tst  if (y_trn.max()+1) < 3 else y_pred_tst
    })
    # save results
    save_res(params,feat_res,trn_res,tst_res,y_trn, y_prob_trn,test_label,y_prob_tst,mode,'GaussianNB')

    return feat_res, trn_res, tst_res, roc_res

def GBDT_ML(params,trn_dat, X_trn, y_trn,X_tst,test_label,y_tst,le,mode):
    feat_res = trn_res = tst_res = roc_res = None
    # constructed model
    gbdt = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],  # 树的数量
        'learning_rate': [0.01, 0.1],  # 学习率
        'max_depth': [3, 5],  # 树的最大深度
        'min_samples_split': [2, 5],  # 分割内部节点所需的最小样本数
        'min_samples_leaf': [1, 2]  # 叶子节点所需的最小样本数
    }
    grid_search = GridSearchCV(gbdt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_trn, y_trn)
    best_gbdt = grid_search.best_estimator_
    # using best model for training
    feature_importances = best_gbdt.feature_importances_
    # extracted sig. feats
    X_trn_sel, important_indices = extract_imp_feat(feature_importances, X_trn)
    # remodelled
    gbdt = GradientBoostingClassifier(**grid_search.best_params_, random_state=42)
    gbdt.fit(X_trn_sel, y_trn)

    if y_trn.max() +1 < 3:
        y_prob_trn = gbdt.predict_proba(X_trn_sel)[:, 1]
        y_prob_tst = gbdt.predict_proba(X_tst[:,important_indices])[:, 1]
        y_pred_trn, y_pred_tst = cal_pred(y_trn,y_prob_trn,y_prob_tst,y_tst)
    else:
        y_prob_trn = gbdt.predict_proba(X_trn_sel)
        y_prob_tst = gbdt.predict_proba(X_tst[:,important_indices])
        y_pred_trn = np.argmax(y_prob_trn, axis=1)
        y_pred_tst = np.argmax(y_prob_tst, axis=1)

    ## save results
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML'))
    # features
    feat_res = pd.DataFrame({
        'Feat': trn_dat.columns,
        'Scores': feature_importances
    })
    # predicted_values
    trn_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_trn),
        'Pred_label': le.inverse_transform(y_pred_trn),
        'Prob': y_prob_trn if (y_trn.max()+1) < 3 else y_pred_trn
    })
    tst_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_tst),
        'Pred_label': le.inverse_transform(y_pred_tst),
        'Prob': y_prob_tst  if (y_trn.max()+1) < 3 else y_pred_tst
    })
    # save results
    save_res(params,feat_res,trn_res,tst_res,y_trn, y_prob_trn,test_label,y_prob_tst,mode,'GBDT')

    return feat_res, trn_res, tst_res, roc_res

def LASSO_ML(params,trn_dat, X_trn, y_trn,X_tst,test_label,y_tst,le,mode):
    feat_res = trn_res = tst_res = roc_res = None
    # constructed model
    lasso = Lasso(random_state=777)
    param_grid = {
        'alpha': np.logspace(-4, 1, 20)  # alpha的范围从10^-4到10^1，共20个点
    }
    grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(X_trn, y_trn)
    best_lasso = grid_search.best_estimator_
    # using best model for training
    feature_importances = best_lasso.coef_
    non_zero_indices = np.where(feature_importances != 0)[0]
    if len(non_zero_indices) > 0:
        X_trn_sel = X_trn[:, non_zero_indices]
    else:
        X_trn_sel = X_trn
        non_zero_indices = np.array(range(len(feature_importances)))

    lasso = Lasso(**grid_search.best_params_, random_state=777)
    lasso.fit(X_trn_sel, y_trn)

    if y_trn.max() +1 < 3:
        y_prob_trn = lasso.predict(X_trn_sel)
        y_prob_tst = lasso.predict(X_tst[:, non_zero_indices])
        y_pred_trn, y_pred_tst = cal_pred(y_trn,y_prob_trn,y_prob_tst,y_tst)
    else:
        y_prob_trn = lasso.predict(X_trn_sel)
        y_prob_tst = lasso.predict(X_tst[:, non_zero_indices])
        y_pred_trn = np.argmax(y_prob_trn, axis=1)
        y_pred_tst = np.argmax(y_prob_tst, axis=1)

    ## save results
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML'))
    # features
    feat_res = pd.DataFrame({
        'Feat': trn_dat.columns,
        'Scores': feature_importances
    })
    # predicted_values
    trn_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_trn),
        'Pred_label': le.inverse_transform(y_pred_trn),
        'Prob': y_prob_trn if (y_trn.max()+1) < 3 else y_pred_trn
    })
    tst_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_tst),
        'Pred_label': le.inverse_transform(y_pred_tst),
        'Prob': y_prob_tst  if (y_trn.max()+1) < 3 else y_pred_tst
    })
    # save results
    save_res(params,feat_res,trn_res,tst_res,y_trn, y_prob_trn,test_label,y_prob_tst,mode,'LASSO')

    return feat_res, trn_res, tst_res, roc_res


def Logit_ML(params, trn_dat, X_trn, y_trn, X_tst, test_label, y_tst, le, mode):
    feat_res = trn_res = tst_res = roc_res = None

    try:
        # Construct model with GridSearchCV
        logreg = LogisticRegression(solver='liblinear')
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2']
        }
        grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_trn, y_trn)
        best_logreg = grid_search.best_estimator_

        # Get feature importances
        feature_importances = np.abs(best_logreg.coef_[0])

        # Try statsmodels Logit with error handling
        try:
            X_trn_with_intercept = sm.add_constant(X_trn)
            logit_model = sm.Logit(y_trn, X_trn_with_intercept).fit(disp=0, maxiter=100)
            coefficients = logit_model.params[1:]
            p_values = logit_model.pvalues[1:]
            indices = np.where(p_values < 0.1)[0]

            if len(indices) > 0:
                X_trn_sel = X_trn[:, indices]
            else:
                # Fallback to sklearn feature importances if no significant features
                X_trn_sel, indices = extract_imp_feat(feature_importances, X_trn)

        except (np.linalg.LinAlgError, Exception) as e:
            print(f"Statsmodels Logit failed: {str(e)}. Using sklearn feature importances instead.")
            X_trn_sel, indices = extract_imp_feat(feature_importances, X_trn)

        # Retrain with selected features
        logreg = LogisticRegression(**grid_search.best_params_, solver='liblinear')
        logreg.fit(X_trn_sel, y_trn)

        # Predictions
        if y_trn.max() + 1 < 3:  # Binary classification
            y_prob_trn = logreg.predict_proba(X_trn_sel)[:, 1]
            y_prob_tst = logreg.predict_proba(X_tst[:, indices])[:, 1]
            y_pred_trn, y_pred_tst = cal_pred(y_trn, y_prob_trn, y_prob_tst, y_tst)
        else:  # Multiclass
            y_prob_trn = logreg.predict_proba(X_trn_sel)
            y_prob_tst = logreg.predict_proba(X_tst[:, indices])
            y_pred_trn = np.argmax(y_prob_trn, axis=1)
            y_pred_tst = np.argmax(y_prob_tst, axis=1)

        # Save results
        if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML')):
            os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML'))

        feat_res = pd.DataFrame({
            'Feat': trn_dat.columns,
            'Scores': feature_importances
        })

        trn_res = pd.DataFrame({
            'True_label': le.inverse_transform(y_trn),
            'Pred_label': le.inverse_transform(y_pred_trn),
            'Prob': y_prob_trn if (y_trn.max() + 1) < 3 else y_pred_trn
        })

        tst_res = pd.DataFrame({
            'True_label': le.inverse_transform(y_tst),
            'Pred_label': le.inverse_transform(y_pred_tst),
            'Prob': y_prob_tst if (y_trn.max() + 1) < 3 else y_pred_tst
        })

        save_res(params, feat_res, trn_res, tst_res, y_trn, y_prob_trn, test_label, y_prob_tst, mode, 'Logit')

    except Exception as e:
        print(f"Error in Logit_ML: {str(e)}")
        # Return empty results or None to continue execution
        return None, None, None, None

    return feat_res, trn_res, tst_res, roc_res

def NN_ML(params,trn_dat, X_trn, y_trn,X_tst,test_label,y_tst,le,mode):
    feat_res = trn_res = tst_res = roc_res = None
    # constructed model
    param_grid = {
        'priors': [None, [0.25, 0.75], [0.75, 0.25]]
    }
    nb = GaussianNB()
    grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_trn, y_trn)
    best_nb = grid_search.best_estimator_

    if y_trn.max() +1 < 3:
        y_prob_trn = best_nb.predict_proba(X_trn)[:, 1]
        y_prob_tst = best_nb.predict_proba(X_tst)[:, 1]
        y_pred_trn, y_pred_tst = cal_pred(y_trn,y_prob_trn,y_prob_tst,y_tst)
    else:
        y_prob_trn = best_nb.predict_proba(X_trn)
        y_prob_tst = best_nb.predict_proba(X_tst)
        y_pred_trn = np.argmax(y_prob_trn, axis=1)
        y_pred_tst = np.argmax(y_prob_tst, axis=1)

    ## save results
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML'))
    # features
    feat_res = pd.DataFrame({
        'Feat': trn_dat.columns,
        'Scores': 'NA'
    })
    # predicted_values
    trn_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_trn),
        'Pred_label': le.inverse_transform(y_pred_trn),
        'Prob': y_prob_trn if (y_trn.max()+1) < 3 else y_pred_trn
    })
    tst_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_tst),
        'Pred_label': le.inverse_transform(y_pred_tst),
        'Prob': y_prob_tst  if (y_trn.max()+1) < 3 else y_pred_tst
    })
    # save results
    save_res(params,feat_res,trn_res,tst_res,y_trn, y_prob_trn,test_label,y_prob_tst,mode,'NN')

    return feat_res, trn_res, tst_res, roc_res

def PLSDA_ML(params,trn_dat, X_trn, y_trn,X_tst,test_label,y_tst,le,mode):
    feat_res = trn_res = tst_res = roc_res = None
    # constructed model
    plsda = PLSRegression()
    param_grid = {
        'n_components': range(1, 17)  # 根据特征数量确定范围
    }
    grid_search = GridSearchCV(estimator=plsda, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(X_trn, y_trn)
    best_params = grid_search.best_params_
    best_plsda = PLSRegression(**best_params)
    best_plsda.fit(X_trn, y_trn)
    feature_importances = compute_VIP(X_trn, y_trn, best_plsda.x_rotations_,
                                      best_plsda.transform(X_trn), best_params['n_components'])
    # extracted sig.feats
    indices = np.where(feature_importances > 1)[0]
    if len(indices) > 0:
        X_trn_sel = X_trn[:, indices]
    else:
        X_trn_sel, indices = extract_imp_feat(feature_importances, X_trn)
    # remodelled
    plsda = PLSRegression(**grid_search.best_params_)
    plsda.fit(X_trn_sel, y_trn)
    if y_trn.max() +1 < 3:
        y_prob_trn = plsda.predict(X_trn_sel).tolist()
        y_prob_tst = plsda.predict(X_tst[:, indices]).tolist()
        y_prob_trn = [item for sublist in y_prob_trn for item in sublist]
        y_prob_tst = [item for sublist in y_prob_tst for item in sublist]
        y_pred_trn, y_pred_tst = cal_pred(y_trn,y_prob_trn,y_prob_tst,y_tst)
    else:
        y_prob_trn = plsda.predict(X_trn_sel).tolist()
        y_prob_tst = plsda.predict(X_tst[:, indices]).tolist()
        y_prob_trn = [item for sublist in y_prob_trn for item in sublist]
        y_prob_tst = [item for sublist in y_prob_tst for item in sublist]
        y_pred_trn = y_prob_trn
        y_pred_tst = y_prob_tst

    ## save results
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML'))
    # features
    feat_res = pd.DataFrame({
        'Feat': trn_dat.columns,
        'Scores': feature_importances
    })
    # predicted_values
    trn_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_trn) if (y_trn.max()+1) < 3 else y_trn,
        'Pred_label': le.inverse_transform(y_pred_trn) if (y_trn.max()+1) < 3 else y_pred_trn,
        'Prob': y_prob_trn
    })
    tst_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_tst) if (y_trn.max()+1) < 3 else y_tst,
        'Pred_label': le.inverse_transform(y_pred_tst) if (y_trn.max()+1) < 3 else y_pred_tst,
        'Prob': y_prob_tst
    })
    # save results
    save_res(params,feat_res,trn_res,tst_res,y_trn, y_prob_trn,test_label,y_prob_tst,mode,'PLSDA')

    return feat_res, trn_res, tst_res, roc_res

def RF_ML(params,trn_dat, X_trn, y_trn,X_tst,test_label,y_tst,le,mode):
    feat_res = trn_res = tst_res = roc_res = None
    # constructed model
    rf = RandomForestClassifier(random_state=777)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0)
    grid_search.fit(X_trn, y_trn)

    # using best model for predicting
    best_model = grid_search.best_estimator_
    feature_importances = best_model.feature_importances_
    # extracted sig. feats
    X_trn_sel, important_indices = extract_imp_feat(feature_importances, X_trn)
    # remodelled
    rf = RandomForestClassifier(**grid_search.best_params_, random_state=777)
    rf.fit(X_trn_sel, y_trn)

    if y_trn.max() +1 < 3:
        y_prob_trn = rf.predict_proba(X_trn_sel)[:, 1]
        y_prob_tst = rf.predict_proba(X_tst[:,important_indices])[:, 1]
        y_pred_trn, y_pred_tst = cal_pred(y_trn,y_prob_trn,y_prob_tst,y_tst)
    else:
        y_prob_trn = rf.predict_proba(X_trn_sel)
        y_prob_tst = rf.predict_proba(X_tst[:,important_indices])
        y_pred_trn = np.argmax(y_prob_trn, axis=1)
        y_pred_tst = np.argmax(y_prob_tst, axis=1)

    ## save results
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML'))
    # features
    feat_res = pd.DataFrame({
        'Feat': trn_dat.columns,
        'Scores': feature_importances
    })
    # predicted_values
    trn_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_trn),
        'Pred_label': le.inverse_transform(y_pred_trn),
        'Prob': y_prob_trn if (y_trn.max()+1) < 3 else y_pred_trn
    })
    tst_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_tst),
        'Pred_label': le.inverse_transform(y_pred_tst),
        'Prob': y_prob_tst  if (y_trn.max()+1) < 3 else y_pred_tst
    })
    # save results
    save_res(params,feat_res,trn_res,tst_res,y_trn, y_prob_trn,test_label,y_prob_tst,mode,'RF')

    return feat_res, trn_res, tst_res, roc_res

def Xgboost_ML(params,trn_dat, X_trn, y_trn,X_tst,test_label,y_tst,le,mode):
    feat_res = trn_res = tst_res = roc_res = None
    # constructed model
    xgb_reg = xgb.XGBRegressor(random_state=777)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_trn, y_trn)
    best_xgb = grid_search.best_estimator_

    # using best model for training
    feature_importances = best_xgb.feature_importances_
    # extracted feats
    X_trn_sel, important_indices = extract_imp_feat(feature_importances, X_trn)
    # remodelled
    xgb_reg = xgb.XGBRegressor(**grid_search.best_params_, random_state=777)
    xgb_reg.fit(X_trn_sel, y_trn)

    if y_trn.max() +1 < 3:
        y_prob_trn = xgb_reg.predict(X_trn_sel)
        y_prob_tst = xgb_reg.predict(X_tst[:,important_indices])
        y_pred_trn, y_pred_tst = cal_pred(y_trn,y_prob_trn,y_prob_tst,y_tst)
    else:
        y_prob_trn = xgb_reg.predict(X_trn_sel)
        y_prob_tst = xgb_reg.predict(X_tst[:,important_indices])
        y_pred_trn = y_prob_trn
        y_pred_tst = y_prob_tst

    ## save results
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML'))
    # features
    feat_res = pd.DataFrame({
        'Feat': trn_dat.columns,
        'Scores': feature_importances
    })
    # predicted_values
    trn_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_trn),
        'Pred_label': le.inverse_transform(y_pred_trn),
        'Prob': y_prob_trn if (y_trn.max()+1) < 3 else y_pred_trn
    })
    tst_res = pd.DataFrame({
        'True_label': le.inverse_transform(y_tst),
        'Pred_label': le.inverse_transform(y_pred_tst),
        'Prob': y_prob_tst  if (y_trn.max()+1) < 3 else y_pred_tst
    })
    # save results
    save_res(params,feat_res,trn_res,tst_res,y_trn, y_prob_trn,test_label,y_prob_tst,mode,'Xgboost')

    return feat_res, trn_res, tst_res, roc_res


def SVM_ML_CV(params, trn_dat, X_trn, y_trn, le, mode):
    # Set up parameter grid for SVM
    param_grid = [
        {
            'kernel': ['linear'],
            'C': np.logspace(-4, 4, 17),  # Wider range with more points
            'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}],  # More class weight options
            'tol': [1e-4, 1e-3, 1e-2],  # Tolerance for stopping criterion
            'max_iter': [1000, 2000, 5000]  # Maximum number of iterations
        },
        # {
        #     'kernel': ['rbf'],
        #     'C': np.logspace(-4, 4, 17),
        #     'gamma': ['scale', 'auto'] + list(np.logspace(-4, 2, 13)),  # More gamma options
        #     'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}],
        #     'tol': [1e-4, 1e-3, 1e-2],
        #     'max_iter': [1000, 2000, 5000],
        #     'shrinking': [True, False]  # Whether to use shrinking heuristic
        # },
        # {
        #     'kernel': ['poly'],
        #     'C': np.logspace(-4, 4, 17),
        #     'gamma': ['scale', 'auto'] + list(np.logspace(-4, 2, 13)),
        #     'degree': [2, 3, 4, 5],  # Extended polynomial degrees
        #     'coef0': [0.0, 0.5, 1.0, 1.5],  # More coefficient options
        #     'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}],
        #     'tol': [1e-4, 1e-3, 1e-2],
        #     'max_iter': [1000, 2000, 5000],
        #     'shrinking': [True, False]
        # },
        {
            'kernel': ['sigmoid'],
            'C': np.logspace(-4, 4, 17),
            'gamma': ['scale', 'auto'] + list(np.logspace(-4, 2, 13)),
            'coef0': [0.0, 0.5, 1.0, 1.5],
            'class_weight': [None, 'balanced', {0: 1, 1: 2}, {0: 1, 1: 3}],
            'tol': [1e-4, 1e-3, 1e-2],
            'max_iter': [1000, 2000, 5000],
            'shrinking': [True, False]
        },
        # # Adding custom kernel (if you have predefined kernel functions)
        # {
        #     'kernel': ['precomputed'],  # For custom kernels
        #     'C': np.logspace(-4, 4, 9),
        #     'class_weight': [None, 'balanced'],
        #     'tol': [1e-4, 1e-3],
        #     'max_iter': [1000, 2000]
        # }
    ]

    # Initialize SVM with probability=True to enable predict_proba
    svm = SVC(probability=True, random_state=777)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=svm,
        param_grid=param_grid,
        cv=params['CV'],
        n_jobs=-1,
        verbose=0,
        scoring='accuracy'  # Can be changed to other metrics
    )

    grid_search.fit(X_trn, y_trn)
    best_svm = grid_search.best_estimator_

    # For linear kernel, we can get feature importances
    if best_svm.kernel == 'linear':
        feature_importances = best_svm.coef_[0]
        # For multi-class, coef_ shape is (n_classes, n_features)
        if len(best_svm.coef_.shape) > 1:
            feature_importances = np.mean(np.abs(best_svm.coef_), axis=0)
    else:
        feature_importances = np.ones(X_trn.shape[1])  # Uniform importance for non-linear

    # Get non-zero features (for linear kernel)
    non_zero_indices = np.where(feature_importances != 0)[0]
    if len(non_zero_indices) > 0 and best_svm.kernel == 'linear':
        X_trn_sel = X_trn[:, non_zero_indices]
    else:
        X_trn_sel = X_trn
        non_zero_indices = np.array(range(len(feature_importances)))

    # Re-train with best parameters on selected features
    best_svm = SVC(**grid_search.best_params_, probability=True, random_state=777)
    best_svm.fit(X_trn_sel, y_trn)

    # Cross-validation evaluation
    kf = KFold(n_splits=params['CV'], shuffle=True, random_state=777)
    y_prob_trn_folds = []
    y_pred_trn_folds = []
    y_true_trn_folds = []
    y_prob_tst_folds = []
    y_pred_tst_folds = []
    y_true_tst_folds = []

    for train_index, test_index in kf.split(X_trn_sel):
        X_trn_fold, X_tst_fold = X_trn_sel[train_index], X_trn_sel[test_index]
        y_trn_fold, y_tst_fold = y_trn[train_index], y_trn[test_index]

        fold_model = SVC(**grid_search.best_params_, probability=True, random_state=777)
        fold_model.fit(X_trn_fold, y_trn_fold)

        # Get probabilities (for binary) or decision function (for multi-class)
        if len(np.unique(y_trn)) == 2:
            y_prob_trn_fold = fold_model.predict_proba(X_trn_fold)[:, 1]
            y_prob_tst_fold = fold_model.predict_proba(X_tst_fold)[:, 1]
        else:
            y_prob_trn_fold = fold_model.predict_proba(X_trn_fold)
            y_prob_tst_fold = fold_model.predict_proba(X_tst_fold)

        y_pred_trn_fold = fold_model.predict(X_trn_fold)
        y_pred_tst_fold = fold_model.predict(X_tst_fold)

        y_true_trn_fold = y_trn_fold.tolist()
        y_true_tst_fold = y_tst_fold.tolist()

        # For binary classification, find optimal threshold
        if len(np.unique(y_trn)) == 2:
            optimal_th, optimal_point = cal_youden(y_trn_fold, y_prob_trn_fold)
            y_pred_trn_fold = [1 if val >= optimal_th else 0 for val in y_prob_trn_fold]
            y_pred_tst_fold = [1 if val >= optimal_th else 0 for val in y_prob_tst_fold]

        y_pred_trn_folds.append(y_pred_trn_fold)
        y_true_trn_folds.append(y_true_trn_fold)
        y_pred_tst_folds.append(y_pred_tst_fold)
        y_true_tst_folds.append(y_true_tst_fold)
        y_prob_trn_folds.append(y_prob_trn_fold)
        y_prob_tst_folds.append(y_prob_tst_fold)

    # Tidy results (using your existing function)
    feature_importances_df, trn_res, tst_res = tidying_CV_res(
        params, feature_importances, trn_dat, y_trn,
        y_pred_trn_folds, y_pred_tst_folds, y_true_trn_folds,
        y_true_tst_folds, y_prob_trn_folds, y_prob_tst_folds, le,
        str('SVM-ML results-') + mode
    )

    if params['ML_Plotting']:
        ROC(le.transform(trn_res['True_label']), le.transform(trn_res['Pred_label']),
            params, 'SVM', mode, 'Train')
        ROC(le.transform(tst_res['True_label']), le.transform(tst_res['Pred_label']),
            params, 'SVM', mode, 'Test')

    return feature_importances_df, trn_res, tst_res

def Boruta_ML_CV(params, trn_dat, X_trn, y_trn, le, mode):
    # constructed model
    param_grid = {
        'n_estimators': [100, 200, 300],  # 决策树的数量
        'max_depth': [None, 5, 10, 20],  # 树的最大深度
        'min_samples_split': [2, 5, 10],  # 分割内部节点所需的最小样本数
        'min_samples_leaf': [1, 2, 4],  # 叶子节点所需的最小样本数
        'max_features': ['auto', 'sqrt', 'log2']  # 寻找最佳分割时要考虑的特征数量
    }
    rf = RandomForestClassifier(n_jobs=-1, class_weight='balanced')
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_trn, y_trn)
    best_rf = grid_search.best_estimator_
    # obtained results of each cv
    kf = KFold(n_splits=params['CV'], shuffle=True, random_state=777)
    y_prob_trn_folds = []
    y_pred_trn_folds = []
    y_true_trn_folds = []
    y_prob_tst_folds = []
    y_pred_tst_folds = []
    y_true_tst_folds = []
    for train_index, test_index in kf.split(X_trn):
        X_trn_fold, X_tst_fold = X_trn[train_index], X_trn[test_index]
        y_trn_fold, y_tst_fold = y_trn[train_index], y_trn[test_index]
        boruta_selector = BorutaPy(best_rf, n_estimators='auto', verbose=0, random_state=1, max_iter=50)
        boruta_selector.fit(X_trn_fold, y_trn_fold)
        selected_features = boruta_selector.support_
        if np.sum(selected_features) > 0:
            X_train_selected = boruta_selector.transform(X_trn_fold)
            X_test_selected = boruta_selector.transform(X_tst_fold)
        else:
            X_train_selected = X_trn_fold
            X_test_selected = X_tst_fold

        rf.fit(X_train_selected, y_trn_fold)
        y_true_trn_fold = y_trn[train_index].tolist()
        y_true_tst_fold = y_trn[test_index].tolist()
        if y_trn.max() + 1 == 2:
            y_prob_trn_fold = rf.predict_proba(X_train_selected)[:,1]
            y_prob_tst_fold = rf.predict_proba(X_test_selected)[:,1]
            optimal_th, optimal_point = cal_youden(y_trn_fold, y_prob_trn_fold)
            y_pred_trn_fold = [1 if val >= optimal_th else 0 for val in y_prob_trn_fold]
            y_pred_tst_fold = [1 if val >= optimal_th else 0 for val in y_prob_tst_fold]
        else:
            y_prob_trn_fold = rf.predict_proba(X_train_selected)
            y_prob_tst_fold = rf.predict_proba(X_test_selected)
            y_pred_trn_fold = np.argmax(y_prob_trn_fold, axis=1)
            y_pred_tst_fold = np.argmax(y_prob_tst_fold, axis=1)

        y_pred_trn_folds.append(y_pred_trn_fold)
        y_true_trn_folds.append(y_true_trn_fold)
        y_pred_tst_folds.append(y_pred_tst_fold)
        y_true_tst_folds.append(y_true_tst_fold)
        y_prob_trn_folds.append(y_prob_trn_fold)
        y_prob_tst_folds.append(y_prob_tst_fold)

    feature_importances_list = []
    for train_index, test_index in kf.split(X_trn):
        boruta_selector = BorutaPy(best_rf, n_estimators='auto', verbose=0, random_state=1, max_iter=50)
        boruta_selector.fit(X_trn[train_index], y_trn[train_index])
        feature_ranks = boruta_selector.ranking_
        feature_ranks = feature_ranks.max() - feature_ranks
        feature_importances_list.append(feature_ranks)
    feature_importances_list = np.mean(feature_importances_list, axis=0)
    # tidying results
    feature_importances_df, trn_res, tst_res = tidying_CV_res(params, feature_importances_list, trn_dat, y_trn,
                                                           y_pred_trn_folds, y_pred_tst_folds, y_true_trn_folds,
                                                           y_true_tst_folds, y_prob_trn_folds,y_prob_tst_folds, le, str('Boruta-ML results-') + mode)
    if params['ML_Plotting']:
        ROC(le.transform(trn_res['True_label']), le.transform(trn_res['Pred_label']), params, 'Boruta', mode, 'Train')
        ROC(le.transform(tst_res['True_label']), le.transform(tst_res['Pred_label']), params, 'Boruta', mode, 'Test')

def GaussianNB_ML_CV(params, trn_dat, X_trn, y_trn,le, mode):
    # constructed and opt model
    param_grid = {
        'priors': [None, [0.25, 0.75], [0.75, 0.25]]
    }
    nb = GaussianNB()
    grid_search = GridSearchCV(estimator=nb, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_trn, y_trn)
    best_nb = grid_search.best_estimator_
    # obtained results of each cv
    kf = KFold(n_splits=params['CV'], shuffle=True, random_state=777)
    y_prob_trn_folds = []
    y_pred_trn_folds = []
    y_true_trn_folds = []
    y_prob_tst_folds = []
    y_pred_tst_folds = []
    y_true_tst_folds = []
    for train_index, test_index in kf.split(X_trn):
        X_trn_fold, X_tst_fold = X_trn[train_index], X_trn[test_index]
        y_trn_fold, y_tst_fold = y_trn[train_index], y_trn[test_index]
        best_nb.fit(X_trn_fold, y_trn_fold)
        y_true_trn_fold = y_trn[train_index].tolist()
        y_true_tst_fold = y_trn[test_index].tolist()
        if y_trn.max() + 1 == 2:
            y_prob_trn_fold = best_nb.predict_proba(X_trn_fold)[:, 1]
            y_prob_tst_fold = best_nb.predict_proba(X_tst_fold)[:, 1]
            optimal_th, optimal_point = cal_youden(y_trn_fold, y_prob_trn_fold)
            y_pred_trn_fold = [1 if val >= optimal_th else 0 for val in y_prob_trn_fold]
            y_pred_tst_fold = [1 if val >= optimal_th else 0 for val in y_prob_tst_fold]
        else:
            y_prob_trn_fold = best_nb.predict_proba(X_trn_fold)
            y_prob_tst_fold = best_nb.predict_proba(X_tst_fold)
            y_pred_trn_fold = np.argmax(y_prob_trn_fold, axis=1)
            y_pred_tst_fold = np.argmax(y_prob_tst_fold, axis=1)


        y_pred_trn_folds.append(y_pred_trn_fold)
        y_true_trn_folds.append(y_true_trn_fold)
        y_pred_tst_folds.append(y_pred_tst_fold)
        y_true_tst_folds.append(y_true_tst_fold)
        y_prob_trn_folds.append(y_prob_trn_fold)
        y_prob_tst_folds.append(y_prob_tst_fold)

    feature_importances_list = ['NA' for _ in range(trn_dat.shape[1])]
    # tidying results
    feature_importances_df, trn_res, tst_res = tidying_CV_res(params, feature_importances_list, trn_dat,y_trn,
                                                              y_pred_trn_folds, y_pred_tst_folds, y_true_trn_folds,
                                                              y_true_tst_folds, y_prob_trn_folds, y_prob_tst_folds, le,
                                                              str('GaussianNB-ML results-') + mode)
    if params['ML_Plotting']:
        ROC(le.transform(trn_res['True_label']), le.transform(trn_res['Pred_label']), params, 'GaussianNB', mode, 'Train')
        ROC(le.transform(tst_res['True_label']), le.transform(tst_res['Pred_label']), params, 'GaussianNB', mode, 'Test')

def GBDT_ML_CV(params, trn_dat, X_trn, y_trn,le, mode):
    # constructed and opt model
    gbdt = GradientBoostingClassifier(random_state=42)
    param_grid = {
        'n_estimators': [100, 200],  # 树的数量
        'learning_rate': [0.01, 0.1],  # 学习率
        'max_depth': [3, 5],  # 树的最大深度
        'min_samples_split': [2, 5],  # 分割内部节点所需的最小样本数
        'min_samples_leaf': [1, 2]  # 叶子节点所需的最小样本数
    }
    grid_search = GridSearchCV(gbdt, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_trn, y_trn)
    best_gbdt = grid_search.best_estimator_
    # using best model for training
    feature_importances = best_gbdt.feature_importances_
    # extracted sig. feats
    X_trn_sel, important_indices = extract_imp_feat(feature_importances, X_trn)
    # remodelled
    gbdt = GradientBoostingClassifier(**grid_search.best_params_, random_state=42)
    gbdt.fit(X_trn_sel, y_trn)
    # obtained results of each cv
    kf = KFold(n_splits=params['CV'], shuffle=True, random_state=777)
    y_prob_trn_folds = []
    y_pred_trn_folds = []
    y_true_trn_folds = []
    y_prob_tst_folds = []
    y_pred_tst_folds = []
    y_true_tst_folds = []
    for train_index, test_index in kf.split(X_trn_sel):
        X_trn_fold, X_tst_fold = X_trn_sel[train_index], X_trn_sel[test_index]
        y_trn_fold, y_tst_fold = y_trn[train_index], y_trn[test_index]
        gbdt.fit(X_trn_fold, y_trn_fold)
        y_true_trn_fold = y_trn[train_index].tolist()
        y_true_tst_fold = y_trn[test_index].tolist()
        if y_trn.max() + 1 ==2:
            y_prob_trn_fold = gbdt.predict_proba(X_trn_fold)[:, 1]
            y_prob_tst_fold = gbdt.predict_proba(X_tst_fold)[:, 1]
            optimal_th, optimal_point = cal_youden(y_trn_fold, y_prob_trn_fold)
            y_pred_trn_fold = [1 if val >= optimal_th else 0 for val in y_prob_trn_fold]
            y_pred_tst_fold = [1 if val >= optimal_th else 0 for val in y_prob_tst_fold]
        else:
            y_prob_trn_fold = gbdt.predict_proba(X_trn_fold)
            y_prob_tst_fold = gbdt.predict_proba(X_tst_fold)
            y_pred_trn_fold = np.argmax(y_prob_trn_fold, axis=1)
            y_pred_tst_fold = np.argmax(y_prob_tst_fold, axis=1)


        y_pred_trn_folds.append(y_pred_trn_fold)
        y_true_trn_folds.append(y_true_trn_fold)
        y_pred_tst_folds.append(y_pred_tst_fold)
        y_true_tst_folds.append(y_true_tst_fold)
        y_prob_trn_folds.append(y_prob_trn_fold)
        y_prob_tst_folds.append(y_prob_tst_fold)

    # tidying results
    feature_importances_df, trn_res, tst_res = tidying_CV_res(params, feature_importances, trn_dat,y_trn,
                                                              y_pred_trn_folds, y_pred_tst_folds, y_true_trn_folds,
                                                              y_true_tst_folds, y_prob_trn_folds, y_prob_tst_folds, le,
                                                              str('GBDT-ML results-') + mode)
    if params['ML_Plotting']:
        ROC(le.transform(trn_res['True_label']), le.transform(trn_res['Pred_label']), params, 'GBDT', mode, 'Train')
        ROC(le.transform(tst_res['True_label']), le.transform(tst_res['Pred_label']), params, 'GBDT', mode, 'Test')

def LASSO_ML_CV(params, trn_dat, X_trn, y_trn,le, mode):
    # constructed and opt model
    lasso = Lasso(random_state=777)
    param_grid = {
        'alpha': np.logspace(-4, 1, 20)  # alpha的范围从10^-4到10^1，共20个点
    }
    grid_search = GridSearchCV(estimator=lasso, param_grid=param_grid, cv=params['CV'], n_jobs=-1, verbose=0)
    grid_search.fit(X_trn, y_trn)
    best_lasso = grid_search.best_estimator_
    # using best model for training
    feature_importances = best_lasso.coef_
    non_zero_indices = np.where(feature_importances != 0)[0]
    if len(non_zero_indices) > 0:
        X_trn_sel = X_trn[:, non_zero_indices]
    else:
        X_trn_sel = X_trn
        non_zero_indices = np.array(range(len(feature_importances)))
    lasso = Lasso(**grid_search.best_params_, random_state=777)
    lasso.fit(X_trn_sel, y_trn)
    # obtained results of each cv
    kf = KFold(n_splits=params['CV'], shuffle=True, random_state=777)
    y_prob_trn_folds = []
    y_pred_trn_folds = []
    y_true_trn_folds = []
    y_prob_tst_folds = []
    y_pred_tst_folds = []
    y_true_tst_folds = []
    for train_index, test_index in kf.split(X_trn_sel):
        X_trn_fold, X_tst_fold = X_trn_sel[train_index], X_trn_sel[test_index]
        y_trn_fold, y_tst_fold = y_trn[train_index], y_trn[test_index]
        lasso.fit(X_trn_fold, y_trn_fold)
        y_prob_trn_fold = lasso.predict(X_trn_fold)
        y_prob_tst_fold = lasso.predict(X_tst_fold)
        y_true_trn_fold = y_trn[train_index].tolist()
        y_true_tst_fold = y_trn[test_index].tolist()
        if y_trn.max() + 1 == 2:
            optimal_th, optimal_point = cal_youden(y_trn_fold, y_prob_trn_fold)
            y_pred_trn_fold = [1 if val >= optimal_th else 0 for val in y_prob_trn_fold]
            y_pred_tst_fold = [1 if val >= optimal_th else 0 for val in y_prob_tst_fold]
        else:
            y_pred_trn_fold = np.argmax(y_prob_trn_fold, axis=1)
            y_pred_tst_fold = np.argmax(y_prob_tst_fold, axis=1)

        y_pred_trn_folds.append(y_pred_trn_fold)
        y_true_trn_folds.append(y_true_trn_fold)
        y_pred_tst_folds.append(y_pred_tst_fold)
        y_true_tst_folds.append(y_true_tst_fold)
        y_prob_trn_folds.append(y_prob_trn_fold)
        y_prob_tst_folds.append(y_prob_tst_fold)

    # tidying results
    feature_importances_df, trn_res, tst_res = tidying_CV_res(params, feature_importances, trn_dat,y_trn,
                                                              y_pred_trn_folds, y_pred_tst_folds, y_true_trn_folds,
                                                              y_true_tst_folds, y_prob_trn_folds, y_prob_tst_folds, le,
                                                              str('LASSO-ML results-') + mode)
    if params['ML_Plotting']:
        ROC(le.transform(trn_res['True_label']), le.transform(trn_res['Pred_label']), params, 'LASSO', mode, 'Train')
        ROC(le.transform(tst_res['True_label']), le.transform(tst_res['Pred_label']), params, 'LASSO', mode, 'Test')

def Logit_ML_CV(params, trn_dat, X_trn, y_trn,le, mode):
    try:
        # constructed and opt model
        logreg = LogisticRegression(solver='liblinear')  # 使用liblinear求解器以支持小数据集
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],  # 正则化强度的倒数
            'penalty': ['l1', 'l2']  # 用于正则化的范数
        }
        grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_trn, y_trn)
        best_logreg = grid_search.best_estimator_
        # using best model for training
        feature_importances = np.abs(best_logreg.coef_[0])
        # extract sig. feats
        X_trn_with_intercept = sm.add_constant(X_trn)
        logit_model = sm.Logit(y_trn, X_trn_with_intercept).fit()
        coefficients = logit_model.params
        p_values = logit_model.pvalues
        p_values_without_intercept = p_values[1:]
        indices = np.where(p_values_without_intercept < 0.1)[0]
        if len(indices) > 0:
            X_trn_sel = X_trn[:, indices]
        else:
            X_trn_sel, indices = extract_imp_feat(feature_importances, X_trn)
        # remodelled
        logreg = LogisticRegression(**grid_search.best_params_, solver='liblinear')
        logreg.fit(X_trn_sel, y_trn)
        # obtained results of each cv
        kf = KFold(n_splits=params['CV'], shuffle=True, random_state=777)
        y_prob_trn_folds = []
        y_pred_trn_folds = []
        y_true_trn_folds = []
        y_prob_tst_folds = []
        y_pred_tst_folds = []
        y_true_tst_folds = []
        for train_index, test_index in kf.split(X_trn_sel):
            X_trn_fold, X_tst_fold = X_trn_sel[train_index], X_trn_sel[test_index]
            y_trn_fold, y_tst_fold = y_trn[train_index], y_trn[test_index]
            logreg.fit(X_trn_fold, y_trn_fold)
            y_true_trn_fold = y_trn[train_index].tolist()
            y_true_tst_fold = y_trn[test_index].tolist()
            if y_trn.max() + 1 == 2:
                y_prob_trn_fold = logreg.predict_proba(X_trn_fold)[:, 1]
                y_prob_tst_fold = logreg.predict_proba(X_tst_fold)[:, 1]
                optimal_th, optimal_point = cal_youden(y_trn_fold, y_prob_trn_fold)
                y_pred_trn_fold = [1 if val >= optimal_th else 0 for val in y_prob_trn_fold]
                y_pred_tst_fold = [1 if val >= optimal_th else 0 for val in y_prob_tst_fold]
            else:
                y_prob_trn_fold = logreg.predict_proba(X_trn_fold)
                y_prob_tst_fold = logreg.predict_proba(X_tst_fold)
                y_pred_trn_fold = np.argmax(y_prob_trn_fold, axis=1)
                y_pred_tst_fold = np.argmax(y_prob_tst_fold, axis=1)

            y_pred_trn_folds.append(y_pred_trn_fold)
            y_true_trn_folds.append(y_true_trn_fold)
            y_pred_tst_folds.append(y_pred_tst_fold)
            y_true_tst_folds.append(y_true_tst_fold)
            y_prob_trn_folds.append(y_prob_trn_fold)
            y_prob_tst_folds.append(y_prob_tst_fold)

        # tidying results
        feature_importances_df, trn_res, tst_res = tidying_CV_res(params, p_values_without_intercept, trn_dat,y_trn,
                                                                  y_pred_trn_folds, y_pred_tst_folds, y_true_trn_folds,
                                                                  y_true_tst_folds, y_prob_trn_folds, y_prob_tst_folds, le,
                                                                  str('logreg-ML results-') + mode)
        if params['ML_Plotting']:
            ROC(le.transform(trn_res['True_label']), le.transform(trn_res['Pred_label']), params, 'Logit', mode,
                'Train')
            ROC(le.transform(tst_res['True_label']), le.transform(tst_res['Pred_label']), params, 'Logit', mode,
                'Test')
    except np.linalg.LinAlgError as e:
        print(f"Singular matrix error encountered: {e}. Skipping Logit model training.")
        return  # 跳过当前模型的训练，继续执行后续代码
    except Exception as e:
        print(f"An unexpected error occurred: {e}. Skipping Logit model training.")
        return  # 跳过当前模型的训练，继续执行后续代码

def NN_ML_CV(params, trn_dat, X_trn, y_trn,le, mode):
    # constructed and opt model
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50)],  # 隐藏层的大小
        'activation': ['tanh', 'relu'],  # 激活函数
        'solver': ['sgd', 'adam'],  # 优化算法
        'alpha': [0.0001, 0.05],  # L2惩罚参数
        'learning_rate': ['constant', 'adaptive'],  # 学习率
    }
    mlp = MLPClassifier(max_iter=300)
    grid_search = GridSearchCV(estimator=mlp, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_trn, y_trn)
    best_mlp = grid_search.best_estimator_
    # obtained results of each cv
    kf = KFold(n_splits=params['CV'], shuffle=True, random_state=777)
    y_prob_trn_folds = []
    y_pred_trn_folds = []
    y_true_trn_folds = []
    y_prob_tst_folds = []
    y_pred_tst_folds = []
    y_true_tst_folds = []
    for train_index, test_index in kf.split(X_trn):
        X_trn_fold, X_tst_fold = X_trn[train_index], X_trn[test_index]
        y_trn_fold, y_tst_fold = y_trn[train_index], y_trn[test_index]
        best_mlp.fit(X_trn_fold, y_trn_fold)
        y_true_trn_fold = y_trn[train_index].tolist()
        y_true_tst_fold = y_trn[test_index].tolist()
        if y_trn.max() + 1 == 2:
            y_prob_trn_fold = best_mlp.predict_proba(X_trn_fold)[:,1]
            y_prob_tst_fold = best_mlp.predict_proba(X_tst_fold)[:,1]
            optimal_th, optimal_point = cal_youden(y_trn_fold, y_prob_trn_fold)
            y_pred_trn_fold = [1 if val >= optimal_th else 0 for val in y_prob_trn_fold]
            y_pred_tst_fold = [1 if val >= optimal_th else 0 for val in y_prob_tst_fold]
        else:
            y_prob_trn_fold = best_mlp.predict_proba(X_trn_fold)
            y_prob_tst_fold = best_mlp.predict_proba(X_tst_fold)
            y_pred_trn_fold = np.argmax(y_prob_trn_fold, axis=1)
            y_pred_tst_fold = np.argmax(y_prob_tst_fold, axis=1)

        y_pred_trn_folds.append(y_pred_trn_fold)
        y_true_trn_folds.append(y_true_trn_fold)
        y_pred_tst_folds.append(y_pred_tst_fold)
        y_true_tst_folds.append(y_true_tst_fold)
        y_prob_trn_folds.append(y_prob_trn_fold)
        y_prob_tst_folds.append(y_prob_tst_fold)

    # tidying results
    feature_importances_list = ['NA' for _ in range(trn_dat.shape[1])]
    feature_importances_df, trn_res, tst_res = tidying_CV_res(params, feature_importances_list, trn_dat,y_trn,
                                                              y_pred_trn_folds, y_pred_tst_folds, y_true_trn_folds,
                                                              y_true_tst_folds, y_prob_trn_folds, y_prob_tst_folds, le,
                                                              str('NN-ML results-') + mode)
    if params['ML_Plotting']:
        ROC(le.transform(trn_res['True_label']), le.transform(trn_res['Pred_label']), params, 'NN', mode, 'Train')
        ROC(le.transform(tst_res['True_label']), le.transform(tst_res['Pred_label']), params, 'NN', mode, 'Test')

def PLSDA_ML_CV(params, trn_dat, X_trn, y_trn,le, mode):
    # constructed and opt model
    plsda = PLSRegression()
    param_grid = {
        'n_components': range(1, 17)  # 根据特征数量确定范围
    }
    grid_search = GridSearchCV(estimator=plsda, param_grid=param_grid, cv=params['CV'], n_jobs=-1, verbose=0)
    grid_search.fit(X_trn, y_trn)
    best_params = grid_search.best_params_
    best_plsda = PLSRegression(**best_params)
    best_plsda.fit(X_trn, y_trn)
    feature_importances = compute_VIP(X_trn, y_trn, best_plsda.x_rotations_,
                                      best_plsda.transform(X_trn), best_params['n_components'])
    # extracted sig.feats
    indices = np.where(feature_importances > 1)[0]
    if len(indices) > 0:
        X_trn_sel = X_trn[:, indices]
    else:
        X_trn_sel, indices = extract_imp_feat(feature_importances, X_trn)
    # obtained results of each cv
    kf = KFold(n_splits=params['CV'], shuffle=True, random_state=777)
    y_prob_trn_folds = []
    y_pred_trn_folds = []
    y_true_trn_folds = []
    y_prob_tst_folds = []
    y_pred_tst_folds = []
    y_true_tst_folds = []
    for train_index, test_index in kf.split(X_trn_sel):
        X_trn_fold, X_tst_fold = X_trn_sel[train_index], X_trn_sel[test_index]
        y_trn_fold, y_tst_fold = y_trn[train_index], y_trn[test_index]
        best_plsda = PLSRegression(**grid_search.best_params_)
        best_plsda.fit(X_trn_fold, y_trn_fold)
        y_true_trn_fold = y_trn[train_index].tolist()
        y_true_tst_fold = y_trn[test_index].tolist()
        if y_trn.max() + 1 == 2:
            y_prob_trn_fold = best_plsda.predict(X_trn_fold)[:, 0]
            y_prob_tst_fold = best_plsda.predict(X_tst_fold)[:, 0]
            optimal_th, optimal_point = cal_youden(y_trn_fold, y_prob_trn_fold)
            y_pred_trn_fold = [1 if val >= optimal_th else 0 for val in y_prob_trn_fold]
            y_pred_tst_fold = [1 if val >= optimal_th else 0 for val in y_prob_tst_fold]
        else:
            y_prob_trn_fold = best_plsda.predict(X_trn_fold)[:, 0]
            y_prob_tst_fold = best_plsda.predict(X_tst_fold)[:, 0]
            y_pred_trn_fold = np.argmax(y_prob_trn_fold, axis=1)
            y_pred_tst_fold = np.argmax(y_prob_tst_fold, axis=1)

        y_pred_trn_folds.append(y_pred_trn_fold)
        y_true_trn_folds.append(y_true_trn_fold)
        y_pred_tst_folds.append(y_pred_tst_fold)
        y_true_tst_folds.append(y_true_tst_fold)
        y_prob_trn_folds.append(y_prob_trn_fold)
        y_prob_tst_folds.append(y_prob_tst_fold)

    # tidying results
    feature_importances_df, trn_res, tst_res = tidying_CV_res(params, feature_importances, trn_dat,y_trn,
                                                              y_pred_trn_folds, y_pred_tst_folds, y_true_trn_folds,
                                                              y_true_tst_folds, y_prob_trn_folds, y_prob_tst_folds, le,
                                                              str('PLSDA-ML results-') + mode)
    if params['ML_Plotting']:
        ROC(le.transform(trn_res['True_label']), le.transform(trn_res['Pred_label']), params, 'PLSDA', mode, 'Train')
        ROC(le.transform(tst_res['True_label']), le.transform(tst_res['Pred_label']), params, 'PLSDA', mode, 'Test')

def RF_ML_CV(params, trn_dat, X_trn, y_trn,le, mode):
    # constructed and opt model
    rf = RandomForestClassifier(random_state=777)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2']
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=params['CV'], n_jobs=-1, verbose=0)
    grid_search.fit(X_trn, y_trn)

    # using best model for predicting
    best_model = grid_search.best_estimator_
    feature_importances = best_model.feature_importances_
    X_trn_sel, important_indices = extract_imp_feat(feature_importances, X_trn)
    # obtained results of each cv
    kf = KFold(n_splits=params['CV'], shuffle=True, random_state=777)
    y_prob_trn_folds = []
    y_pred_trn_folds = []
    y_true_trn_folds = []
    y_prob_tst_folds = []
    y_pred_tst_folds = []
    y_true_tst_folds = []
    for train_index, test_index in kf.split(X_trn_sel):
        X_trn_fold, X_tst_fold = X_trn_sel[train_index], X_trn_sel[test_index]
        y_trn_fold, y_tst_fold = y_trn[train_index], y_trn[test_index]
        best_model.fit(X_trn_fold, y_trn_fold)
        y_true_trn_fold = y_trn[train_index].tolist()
        y_true_tst_fold = y_trn[test_index].tolist()
        if y_trn.max() + 1 == 2:
            y_prob_trn_fold = best_model.predict_proba(X_trn_fold)[:,1]
            y_prob_tst_fold = best_model.predict_proba(X_tst_fold)[:,1]
            optimal_th, optimal_point = cal_youden(y_trn_fold, y_prob_trn_fold)
            y_pred_trn_fold = [1 if val >= optimal_th else 0 for val in y_prob_trn_fold]
            y_pred_tst_fold = [1 if val >= optimal_th else 0 for val in y_prob_tst_fold]
        else:
            y_prob_trn_fold = best_model.predict_proba(X_trn_fold)
            y_prob_tst_fold = best_model.predict_proba(X_tst_fold)
            y_pred_trn_fold = np.argmax(y_prob_trn_fold, axis=1)
            y_pred_tst_fold = np.argmax(y_prob_tst_fold, axis=1)

        y_pred_trn_folds.append(y_pred_trn_fold)
        y_true_trn_folds.append(y_true_trn_fold)
        y_pred_tst_folds.append(y_pred_tst_fold)
        y_true_tst_folds.append(y_true_tst_fold)
        y_prob_trn_folds.append(y_prob_trn_fold)
        y_prob_tst_folds.append(y_prob_tst_fold)

    # tidying results
    feature_importances_df, trn_res, tst_res = tidying_CV_res(params, feature_importances, trn_dat,y_trn,
                                                              y_pred_trn_folds, y_pred_tst_folds, y_true_trn_folds,
                                                              y_true_tst_folds, y_prob_trn_folds, y_prob_tst_folds, le,
                                                              str('RF-ML results-') + mode)
    if params['ML_Plotting']:
        ROC(le.transform(trn_res['True_label']), le.transform(trn_res['Pred_label']), params, 'RF', mode, 'Train')
        ROC(le.transform(tst_res['True_label']), le.transform(tst_res['Pred_label']), params, 'RF', mode, 'Test')

def Xgboost_ML_CV(params, trn_dat, X_trn, y_trn,le, mode):
    # constructed and opt model
    xgb_reg = xgb.XGBRegressor(random_state=777)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 4, 5],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    grid_search = GridSearchCV(estimator=xgb_reg, param_grid=param_grid, cv=5, n_jobs=-1, verbose=0,
                               scoring='neg_mean_squared_error')
    grid_search.fit(X_trn, y_trn)
    best_xgb = grid_search.best_estimator_
    # using best model for training
    feature_importances = best_xgb.feature_importances_
    # extracted feats
    X_trn_sel, important_indices = extract_imp_feat(feature_importances, X_trn)
    # remodelled
    xgb_reg = xgb.XGBRegressor(**grid_search.best_params_, random_state=777)
    xgb_reg.fit(X_trn_sel, y_trn)
    # obtained results of each cv
    kf = KFold(n_splits=params['CV'], shuffle=True, random_state=777)
    y_prob_trn_folds = []
    y_pred_trn_folds = []
    y_true_trn_folds = []
    y_prob_tst_folds = []
    y_pred_tst_folds = []
    y_true_tst_folds = []
    for train_index, test_index in kf.split(X_trn_sel):
        X_trn_fold, X_tst_fold = X_trn_sel[train_index], X_trn_sel[test_index]
        y_trn_fold, y_tst_fold = y_trn[train_index], y_trn[test_index]
        xgb_reg.fit(X_trn_fold, y_trn_fold)
        y_prob_trn_fold = xgb_reg.predict(X_trn_fold)
        y_prob_tst_fold = xgb_reg.predict(X_tst_fold)
        y_true_trn_fold = y_trn[train_index].tolist()
        y_true_tst_fold = y_trn[test_index].tolist()
        if y_trn.max() + 1 == 2:
            optimal_th, optimal_point = cal_youden(y_trn_fold, y_prob_trn_fold)
            y_pred_trn_fold = [1 if val >= optimal_th else 0 for val in y_prob_trn_fold]
            y_pred_tst_fold = [1 if val >= optimal_th else 0 for val in y_prob_tst_fold]
        else:
            y_pred_trn_fold = y_prob_trn_fold.tolist()
            y_pred_tst_fold = y_prob_tst_fold.tolist()

        y_pred_trn_folds.append(y_pred_trn_fold)
        y_true_trn_folds.append(y_true_trn_fold)
        y_pred_tst_folds.append(y_pred_tst_fold)
        y_true_tst_folds.append(y_true_tst_fold)
        y_prob_trn_folds.append(y_prob_trn_fold)
        y_prob_tst_folds.append(y_prob_tst_fold)

    # tidying results
    feature_importances_df, trn_res, tst_res = tidying_CV_res(params, feature_importances, trn_dat,y_trn,
                                                              y_pred_trn_folds, y_pred_tst_folds, y_true_trn_folds,
                                                              y_true_tst_folds, y_prob_trn_folds, y_prob_tst_folds, le,
                                                              str('Xgboost-ML results-') + mode)
    if params['ML_Plotting']:
        ROC(le.transform(trn_res['True_label']), le.transform(trn_res['Pred_label']), params, 'Xgboost', mode, 'Train')
        ROC(le.transform(tst_res['True_label']), le.transform(tst_res['Pred_label']), params, 'Xgboost', mode, 'Test')

def tidying_CV_res(params, feature_importances, trn_dat, y_trn, y_pred_trn_folds, y_pred_tst_folds, y_true_trn_folds,
                y_true_tst_folds, y_prob_trn_folds,y_prob_tst_folds, le, mode):
    if not os.path.exists(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML')):
        os.makedirs(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML'))
    # features
    feature_importances_df = pd.DataFrame({
            'Feat': trn_dat.columns,
            'Scores': feature_importances
        })
    # predicted_values
    trn_pred_Folds = []
    tst_pred_Folds = []
    for i in range(len(y_pred_trn_folds)):
        trn_pred_Fold = [i for _ in range(len(y_pred_trn_folds[i]))]
        tst_pred_Fold = [i for _ in range(len(y_pred_tst_folds[i]))]
        trn_pred_Folds.append(trn_pred_Fold)
        tst_pred_Folds.append(tst_pred_Fold)

    predictions = [item for sublist in y_pred_trn_folds for item in sublist]
    prob = [item for sublist in y_prob_trn_folds for item in sublist]
    True_Labels = [item for sublist in y_true_trn_folds for item in sublist]
    tst_predictions = [item for sublist in y_pred_tst_folds for item in sublist]
    tst_prob = [item for sublist in y_prob_tst_folds for item in sublist]
    tst_True_Labels = [item for sublist in y_true_tst_folds for item in sublist]
    Folds = [item for sublist in trn_pred_Folds for item in sublist]
    tst_Folds = [item for sublist in tst_pred_Folds for item in sublist]

    trn_res = pd.DataFrame({
        'Fold': Folds,
        'Prob': prob,
        'Pred_label': predictions,
        'True_label': True_Labels
    })
    tst_res = pd.DataFrame({
        'Fold': tst_Folds,
        'Prob': tst_prob,
        'Pred_label': tst_predictions,
        'True_label': tst_True_Labels
    })
    if 'PLSDA' not in mode and 'Xgboost' not in mode:
        trn_res['True_label'] = [le.inverse_transform([label])[0] for label in trn_res['True_label']]
        tst_res['True_label'] = [le.inverse_transform([label])[0] for label in tst_res['True_label']]
        trn_res['Pred_label'] = [le.inverse_transform([label])[0] for label in trn_res['Pred_label']]
        tst_res['Pred_label'] = [le.inverse_transform([label])[0] for label in tst_res['Pred_label']]
    else:
        trn_res['Pred_label'] = le.inverse_transform(trn_res['Pred_label'].values)
        tst_res['Pred_label'] = le.inverse_transform(tst_res['Pred_label'].values)
        trn_res['True_label'] = le.inverse_transform(trn_res['True_label'].values)
        tst_res['True_label'] = le.inverse_transform(tst_res['True_label'].values)
        # save results
    feature_importances_df.to_csv(
        os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML', f'Features Scores-{mode}.csv'), index=None)
    trn_res.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML', f'Trn dat-{mode}.csv'), index=None)
    tst_res.to_csv(os.path.join(params['Parent_FilePath'], params['project_name'], 'Results/ML', f'Test dat-{mode}.csv'), index=None)

    return feature_importances_df, trn_res, tst_res

def cal_pred(y_trn,y_prob_trn,y_prob_tst,y_tst):
    if y_trn.max()+1 <= 2:
        optimal_th, optimal_point = cal_youden(y_trn, y_prob_trn)
        y_pred_trn = [1 if val >= optimal_th else 0 for val in y_prob_trn]
        y_pred_tst = [1 if val >= optimal_th else 0 for val in y_prob_tst]
    else:
        # binary_y_trn_0_vs_12 = (y_trn != 0).astype(int)
        # optimal_th_0_vs_12, optimal_point_0_vs_12 = cal_youden(binary_y_trn_0_vs_12, y_prob_trn)
        # binary_y_trn_1_vs_2 = (y_trn == 2).astype(int)
        # optimal_th_1_vs_2, optimal_point_1_vs_2 = cal_youden(binary_y_trn_1_vs_2, y_prob_trn)
        # y_pred_trn = np.ones(len(y_prob_trn), dtype=int)
        # if optimal_th_0_vs_12 > optimal_th_1_vs_2:
        #     y_pred_trn[y_prob_trn >= optimal_th_0_vs_12] = 2
        #     y_pred_trn[(y_prob_trn >= optimal_th_1_vs_2) & (y_prob_trn < optimal_th_0_vs_12)] = 1
        #     y_pred_trn[y_prob_trn < optimal_th_1_vs_2] = 0
        # else:
        #     y_pred_trn[y_prob_trn >= optimal_th_1_vs_2] = 2
        #     y_pred_trn[(y_prob_trn >= optimal_th_0_vs_12) & (y_prob_trn < optimal_th_1_vs_2)] = 1
        #     y_pred_trn[y_prob_trn < optimal_th_0_vs_12] = 0
        #
        # binary_y_tst_0_vs_12 = (y_tst != 0).astype(int)
        # optimal_th_0_vs_12, optimal_point_0_vs_12 = cal_youden(binary_y_tst_0_vs_12, y_prob_tst)
        # binary_y_tst_1_vs_2 = (y_tst == 2).astype(int)
        # optimal_th_1_vs_2, optimal_point_1_vs_2 = cal_youden(binary_y_tst_1_vs_2, y_prob_tst)
        # y_pred_tst = np.ones(len(y_prob_tst), dtype=int)
        # if optimal_th_0_vs_12 > optimal_th_1_vs_2:
        #     y_pred_tst[y_prob_tst >= optimal_th_0_vs_12] = 2
        #     y_pred_tst[(y_prob_tst >= optimal_th_1_vs_2) & (y_prob_tst < optimal_th_0_vs_12)] = 1
        #     y_pred_tst[y_prob_tst < optimal_th_1_vs_2] = 0
        # else:
        #     y_pred_tst[y_prob_tst >= optimal_th_1_vs_2] = 2
        #     y_pred_tst[(y_prob_tst >= optimal_th_0_vs_12) & (y_prob_tst < optimal_th_1_vs_2)] = 1
        #     y_pred_tst[y_prob_tst < optimal_th_0_vs_12] = 0
        y_pred_trn = y_prob_trn
        y_pred_tst = y_prob_tst

    return y_pred_trn,y_pred_tst


















