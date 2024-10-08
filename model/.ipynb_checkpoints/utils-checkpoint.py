# Standard packages
# %matplotlib inline
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import tensorflow.compat.v1 as tf
# tf.disable_eager_execution()
tf.logging.set_verbosity(tf.logging.ERROR)
import seaborn as sns
import random
from tqdm import tqdm
from numpy import mean
from numpy import std
from IPython.display import Markdown, display

# Plotting 
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Sklearn
# from sklearn.cluster import KMeans
# from sklearn.pipeline import Pipeline
from sklearn.compose import make_column_transformer
from sklearn.model_selection import GridSearchCV, train_test_split, ParameterGrid
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import StratifiedKFold
## accuracy metrics
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score, matthews_corrcoef
from sklearn.metrics import recall_score, precision_score # for fairness metrics (calculate diff)
## base estimators
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# Fairness metrics
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.metrics import ClassificationMetric
from aif360.sklearn.metrics import difference as aif_difference
# from aif360.sklearn.metrics import specificity_score, false_omission_rate_error
from aif360.sklearn.metrics import specificity_score
from aif360.sklearn.metrics import statistical_parity_difference, average_odds_difference, \
                                   equal_opportunity_difference, average_odds_error
from aif360.sklearn.metrics import consistency_score, generalized_entropy_index, theil_index

# Explainers
from aif360.explainers import MetricTextExplainer

# Scalers
from sklearn.preprocessing import StandardScaler

# Classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# Bias mitigation techniques
# from aif360.algorithms.preprocessing import Reweighing
# from aif360.algorithms.preprocessing import OptimPreproc
# from aif360.algorithms.preprocessing import DisparateImpactRemover
# from aif360.algorithms.preprocessing import LFR
from aif360.sklearn.preprocessing import Reweighing, ReweighingMeta, LearnedFairRepresentations
from aif360.sklearn.inprocessing import AdversarialDebiasing, ExponentiatedGradientReduction
from aif360.sklearn.postprocessing import CalibratedEqualizedOdds, RejectOptionClassifier


import math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

# from tabtransformertf.models.tabtransformer import TabTransformer
# from tabtransformertf.utils.preprocessing import df_to_dataset, build_categorical_prep

# set seed to reproduce code for TabTransformer
random_state = 1234
tf.random.set_seed(random_state)
from numpy.random import seed
seed(random_state)

# -----------------------------------------------------------------------------------
# taken from AIF360/aif360/sklearn/metrics/metrics.py due to import error
from sklearn.metrics._classification import _prf_divide, _check_zero_division
from sklearn.metrics import multilabel_confusion_matrix

def false_omission_rate_error(y_true, y_pred, *, pos_label=1, sample_weight=None,
                      zero_division='warn'):
    """Compute the false omission rate.
    Args:
        y_true (array-like): Ground truth (correct) target values.
        y_pred (array-like): Estimated targets as returned by a classifier.
        pos_label (scalar, optional): The label of the positive class.
        sample_weight (array-like, optional): Sample weights.
        zero_division ('warn', 0 or 1): Sets the value to return when there is a
            zero division. If set to “warn”, this acts as 0, but warnings are
            also raised.
    """
    _check_zero_division(zero_division)
    MCM = multilabel_confusion_matrix(y_true, y_pred, labels=[pos_label],
                                      sample_weight=sample_weight)
    tn, fn = MCM[:, 0, 0], MCM[:, 1, 0]
    negs = tn + fn
    return _prf_divide(fn, negs, 'false omission rate', 'predicted negative', None,
                       ('false omission rate',), zero_division).item()
# -----------------------------------------------------------------------------------


accuracy_metrics = ['acc_score', 'bacc_score', 'f1_score', 'auc_score', 'mcc_score','norm_mcc_score']
group_fairness = ['spd_score', 'aod_score', 'eod_score','ford_score','ppvd_score']
individual_fairness = ['(1-consistency_score)','gei_score','ti_score']
fairness_metrics = group_fairness+individual_fairness

# bm_category = {'None':'None', 'RW':'pre', 'LFR_pre':'pre',
#                'LFR_in':'in', 'AD':'in', 'EGR':'in',
#                'ROC':'post', 'CEO':'post'}

bm_category = {'PRE':['RW', 'LFR_pre'],
               'IN': ['LFR_in','AD','EGR'],
               'POST':['ROC','CEO']}

def store_metrics(y_test, y_pred, X_test, pred_prob, thres_dict, BM_name, threshold, priv_group, pos_label, prot_attr):
    """Returns a dictionary with all interested accuracy and fairness metrics.
        Args:
            y_test (array-like): true labels from test set.
            y_pred (array-like): predicted labels for test set.
            thres_dict (dict): dictionary that stores all info.
            threshold (np.float): given threshold used to obtain y_pred.
        Returns:
            dict: `thres_dict`
    """
    # evaluate model performance for each split
    # --------------------------------------- Accuracy Metrics ---------------------------------------
    thres_dict[BM_name][threshold]['acc_score'] += [accuracy_score(y_test, y_pred)]
    thres_dict[BM_name][threshold]['bacc_score'] += [balanced_accuracy_score(y_test, y_pred)]
    thres_dict[BM_name][threshold]['f1_score'] += [f1_score(y_test, y_pred)]
    thres_dict[BM_name][threshold]['auc_score'] += [roc_auc_score(y_test, pred_prob)]
    thres_dict[BM_name][threshold]['mcc_score'] += [matthews_corrcoef(y_test, y_pred)]
    thres_dict[BM_name][threshold]['norm_mcc_score'] += [0.5*(matthews_corrcoef(y_test, y_pred)+1)]
    # ------------------------------------- Group Fairness Metrics ------------------------------------
    thres_dict[BM_name][threshold]['spd_score'] += [statistical_parity_difference(y_test, y_pred, prot_attr=prot_attr,
                                                                                  priv_group=priv_group, pos_label=pos_label)]
    thres_dict[BM_name][threshold]['aod_score'] += [average_odds_difference(y_test, y_pred, prot_attr=prot_attr,
                                                                            priv_group=priv_group, pos_label=pos_label)]
    thres_dict[BM_name][threshold]['eod_score'] += [equal_opportunity_difference(y_test, y_pred, prot_attr=prot_attr,
                                                                                 priv_group=priv_group, pos_label=pos_label)]
    thres_dict[BM_name][threshold]['ford_score'] += [aif_difference(false_omission_rate_error, y_test, y_pred,
                                                                    prot_attr=prot_attr, priv_group=priv_group, 
                                                                    pos_label=pos_label)]
    thres_dict[BM_name][threshold]['ppvd_score'] += [aif_difference(precision_score, y_test, y_pred, prot_attr=prot_attr,
                                                                    priv_group=priv_group, pos_label=pos_label)]
    # ---------------------------------- Individual Fairness Metrics ----------------------------------
    try: thres_dict[BM_name][threshold]['(1-consistency_score)'] += [1-consistency_score(X_test, y_pred)]
    except: 
        # get dummies for categorical features in X_test for calculating consistency score
        CATEGORICAL_FEATURES = X_test.select_dtypes(exclude=np.number).columns
        X_test = pd.get_dummies(X_test, columns = CATEGORICAL_FEATURES)
        thres_dict[BM_name][threshold]['(1-consistency_score)'] += [1-consistency_score(X_test, y_pred)]

    thres_dict[BM_name][threshold]['gei_score'] += [generalized_entropy_index(b=y_pred-y_test+1)] # ref: speicher_unified_2018
    thres_dict[BM_name][threshold]['ti_score'] += [theil_index(b=y_pred-y_test+1)]

    return thres_dict

def get_avg_metrics(thres_dict):
    """Returns the average of all cv splits from the same model setting (hyperparameter and threshold).
    Args:
        thres_dict (dict): the dictionary with all info on each cv split.
    Returns:
        dict: `final_metrics`
    """ 
    import copy
    # calculate the average for each metrics from all splits
    avg_metrics = copy.deepcopy(thres_dict)
    for BM in avg_metrics.keys():
        for threshold in avg_metrics[BM].keys(): 
            average_list = {}
            for metric in avg_metrics[BM][threshold].keys():
                average_list['avg_%s'%metric] = mean(avg_metrics[BM][threshold][metric])
            avg_metrics[BM][threshold]['average'] = average_list
    return avg_metrics

def get_output_table(all_metrics, base, scoring):
    """Returns the output table from all param_grid.
    Args:
        all_metrics (dict): the final dictionary with info from all param_grid.
        base (str): the name of the base estimator that is shown in the output table.
    """ 

    output_table = pd.DataFrame()
    for model in all_metrics.keys():
        all_metrics[model]['parameters']['hparam'].pop('random_state', None)
        table_cv = pd.DataFrame(all_metrics[model]['metrics']['average'], index=[0])
        table_cv.insert(0, 'base_estimator', base)
        table_cv.insert(1, 'param', str(all_metrics[model]['parameters']['hparam']))
        table_cv.insert(2, 'Bias_Mitigation', str(all_metrics[model]['parameters']['Bias_Mitigation']))
        table_cv.insert(3, 'threshold', all_metrics[model]['parameters']['threshold'])
        # table_cv[['base_estimator','param']] = np.where(table_cv['Bias_Mitigation'].isin(bm_category['IN'])&\
        #                                                 (table_cv['Bias_Mitigation']!='EGR'), 
        #                                                 '', table_cv[['base_estimator','param']])
        output_table = pd.concat([output_table, table_cv]).reset_index(drop=True)
    # find "best" model
    acc_metric = 'avg_'+scoring[0].lower()+'_score'
    fair_metric = 'avg_'+scoring[1].lower()+'_score'
    w_acc = scoring[2]
    w_fair = scoring[3]
    acc_cost = 1-output_table[acc_metric]
    fair_cost = abs(output_table[fair_metric])

    output_table['cost'] = w_acc*acc_cost + w_fair*fair_cost
    return output_table

def style_table(df):
    """Returs the output table with highlight on the best metrics
    Args:
        df (DataFrame): the output table to be styled
    """
    avg_accuracy_metrics = ['avg_'+col for col in accuracy_metrics]
    avg_fairness_metrics = ['avg_'+col for col in fairness_metrics]

    best_index = np.argmin(df.cost)
    df = df.style.highlight_max(subset=avg_accuracy_metrics,color='lightgreen')\
                   .apply(lambda s:['background: yellow' if abs(cell)==min(abs(s)) else '' for cell in s],
                          subset=avg_fairness_metrics)\
                   .highlight_min(subset=['cost'],color='lightblue')\
                   .apply(lambda s:['font-weight: bold' if v == s.iloc[best_index] else '' for v in s])
    
    return df

def merge_dictionary_list(dict_list):
    result_dict = {}
    try:
        for d in dict_list:
            for BM in d:
                if BM not in result_dict:
                    result_dict[BM] = {}
                for thres in d[BM]:
                    if thres not in result_dict[BM]:
                        result_dict[BM][thres] = {}
                    for metric in d[BM][thres]:
                        if metric not in result_dict[BM][thres]:
                            result_dict[BM][thres][metric] = []
                        result_dict[BM][thres][metric].append(d[BM][thres][metric].pop())
    except: pass
    return result_dict

def get_num_cat_col(X_train, X_test, y_train , y_test):
    LABEL = y_train.name
    train_data = pd.concat([X_train,y_train], axis=1)
    try: test_data = pd.concat([X_test, y_test], axis=1)
    except: pass
    NUMERIC_FEATURES = train_data.drop(LABEL, axis=1).select_dtypes(include=np.number).columns
    CATEGORICAL_FEATURES = train_data.drop(LABEL, axis=1).select_dtypes(exclude=np.number).columns
    
    return train_data, test_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES
    
def tf_dataset_for_TrainValTest(train_data, test_data, CATEGORICAL_FEATURES, NUMERIC_FEATURES):
    LABEL = train_data.columns[-1]
    FEATURES = list(NUMERIC_FEATURES) + list(CATEGORICAL_FEATURES)
    # format dataset
    for dataset in [train_data, test_data]:
        dataset[LABEL] = dataset[LABEL].apply(lambda x: int(x)) 
        dataset[CATEGORICAL_FEATURES] = dataset[CATEGORICAL_FEATURES].astype(str)
        dataset[NUMERIC_FEATURES] = dataset[NUMERIC_FEATURES].astype(float)
    # Train/val split
    train, val = train_test_split(train_data, test_size=0.2, random_state=random_state)
    # Category preprocessing layers
    category_prep_layers = build_categorical_prep(train, CATEGORICAL_FEATURES)
    # To TF Dataset
    train_dataset = df_to_dataset(train[FEATURES + [LABEL]], LABEL)
    val_dataset = df_to_dataset(val[FEATURES + [LABEL]], LABEL, shuffle=False)  # No shuffle
    test_dataset = df_to_dataset(test_data[FEATURES], shuffle=False) # No target, no shuffle
    
    return train, val, train_dataset, val_dataset, test_dataset
