import warnings

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


# Set plot font
plt.rcParams.update({'font.family':'serif'})
plt.rcParams.update({'font.serif':'Times New Roman'})


import math
import numpy as np
import pandas as pd
# import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_addons as tfa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score
from tensorflow.keras.callbacks import EarlyStopping

from tabtransformertf.models.tabtransformer import TabTransformer
from tabtransformertf.utils.preprocessing import df_to_dataset, build_categorical_prep

# set seed to reproduce code for TabTransformer
random_state = 1234
tf.random.set_random_seed(random_state)
from numpy.random import seed
seed(random_state)

# from sklearn.utils.parallel import delayed, Parallel
import multiprocessing #original multiprocessing
import multiprocess as mp #better multiprocessing for python
from itertools import repeat
from multiprocessing import set_start_method
set_start_method("spawn")
from multiprocessing import get_context

from utils import*
class skf_model():
    """
    stratified k-fold model class, to implement the grid-search framework 
    with k-fold to ensure performance stability.
    """
    def __init__(self, prot_attr, pos_label, priv_group, cv, random_state, n_jobs):
        self.prot_attr = prot_attr
        self.pos_label = pos_label
        self.priv_group = priv_group
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        
    def run_search(self, train_index, test_index, X, y, base, param, BM_arr, thres_arr, thres_dict):
        if base=='TabTrans':
            LABEL = y.name
            X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train , y_test = y[train_index] , y[test_index]
            train_data,test_data,NUMERIC_FEATURES,CATEGORICAL_FEATURES = get_num_cat_col(X_train,X_test,y_train,y_test)
            train, val, train_dataset, val_dataset, test_dataset = tf_dataset_for_TrainValTest(train_data, test_data,
                                                                                               CATEGORICAL_FEATURES, 
                                                                                               NUMERIC_FEATURES)
            category_prep_layers = build_categorical_prep(train, CATEGORICAL_FEATURES)
        else:
            categorical_cols = list(X.select_dtypes(exclude=[np.number]).columns)
            X = pd.get_dummies(X, columns = categorical_cols)

            X_train, X_test = X.iloc[train_index,:],X.iloc[test_index,:]
            y_train , y_test = y[train_index] , y[test_index]

            # normalize data features, fit only on training data to avoid data leakage
            scaler = StandardScaler()
            scaler.fit(X_train[X_train.columns])
            X_train[X_train.columns] = scaler.transform(X_train[X_train.columns])
            X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])

        # assign base estimator
        if base not in ['NB','TabTrans']:
            param.update({'random_state': self.random_state})
        else: pass

        if base=='LR':
            base_estimator=LogisticRegression(**param)
        elif base=='RF':
            base_estimator=RandomForestClassifier(**param)
        elif base=='GB':
            base_estimator=GradientBoostingClassifier(**param)
        elif base=='SVM':
            param.update({'probability':True})
            base_estimator=SVC(**param)
        elif base=='NB':
            base_estimator=GaussianNB(**param)
        elif base=='TabTrans':
            TabTrans_param = {'numerical_features': NUMERIC_FEATURES,
                              'categorical_features': CATEGORICAL_FEATURES,
                              'categorical_lookup': category_prep_layers,
                              'embedding_dim': 32, 'out_dim': 1,
                              'out_activation': 'sigmoid','depth': 4,'heads': 8,
                              'attn_dropout': 0.2, 'ff_dropout': 0.2,
                              'mlp_hidden_factors': [2, 4], 'use_column_embedding': True}
            base_estimator=TabTransformer(**TabTrans_param)

            print('-'*90)
            print(param)
            print('-'*90)
            LEARNING_RATE = param['learing_rate'] if 'learing_rate' in param else 0.0001
            WEIGHT_DECAY = param['weight_decay'] if 'weight_decay' in param else 0.0001
            NUM_EPOCHS = param['epochs'] if 'epochs' in param else 10

            optimizer = tfa.optimizers.AdamW(learning_rate=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
            # base_estimator.compile(optimizer = optimizer,
            #                        loss = tf.keras.losses.BinaryCrossentropy(),
            #                        metrics= [tf.keras.metrics.AUC(name="PR AUC", curve='PR')],)
            early = EarlyStopping(monitor="val_loss", mode="min", patience=10, restore_best_weights=True)
            callback_list = [early]

        # assign bias mitigation
        BM_dict = {None: base_estimator, 
                   # pre-processing
                   # 'RW': ReweighingMeta(estimator=base_estimator), 
                   'RW': Reweighing(prot_attr=self.prot_attr), 
                   'LFR_pre': LearnedFairRepresentations(prot_attr=self.prot_attr, random_state=self.random_state),
                   # in-processing
                   'LFR_in': LearnedFairRepresentations(prot_attr=self.prot_attr, random_state=self.random_state),
                   'AD': AdversarialDebiasing(prot_attr=self.prot_attr, random_state=self.random_state),
                   'EGR': ExponentiatedGradientReduction(prot_attr=self.prot_attr, estimator=base_estimator,
                                                         constraints="EqualizedOdds"),
                   # post-processing
                   'ROC': base_estimator,
                   'CEO': base_estimator} 

        for BM in BM_arr:
            print('running Bias Mitigation {}'.format(BM))
            if (base=='TabTrans')&((BM=='LFR_pre')|(BM=='EGR')): pass
            else:
                BM_name = str(BM)
                BMs = BM.split('+') if BM!=None else [None]
                model = BM_dict[BMs[0]]
                # fit
                if 'RW' in BMs:
                    # model.fit(X_train, y_train)
                    sample_weight = None 
                    if base!='TabTrans': 
                        X_train, sample_weight = model.fit_transform(X_train, y_train, sample_weight)
                        base_estimator.fit(X_train, y_train, sample_weight)
                    else: 
                        # first reweigh training and validating data
                        train_idx, val_idx = train.index, val.index
                        tmp_data = pd.concat([train, val])
                        tmp_data, sample_weight = model.fit_transform(tmp_data.drop(LABEL, axis=1),
                                                                      tmp_data[LABEL], sample_weight)
                        train, val = pd.concat([tmp_data.loc[train_idx], train.loc[train_idx,LABEL]], axis=1),\
                                     pd.concat([tmp_data.loc[val_idx], val.loc[val_idx,LABEL]], axis=1)
                        # then convert them to tf_datasets
                        train_dataset = df_to_dataset(train, target=y_train.name)
                        val_dataset = df_to_dataset(val, target=y_train.name, shuffle=False)
                        # finally re-build category layer and fit
                        category_prep_layers = build_categorical_prep(train, CATEGORICAL_FEATURES)

                        base_estimator.compile(optimizer = optimizer,
                                               loss = tf.keras.losses.BinaryCrossentropy(),
                                               weighted_metrics = [tf.keras.metrics.AUC(name="PR AUC", curve='PR')],)
                        base_estimator.fit(train_dataset, epochs=NUM_EPOCHS, 
                                      validation_data=val_dataset, callbacks=callback_list)
                    model = base_estimator # now the base estimator is fit on reweighed training data
                elif BM=='LFR_pre':
                    if base!='TabTrans': 
                        model.fit(X_train, y_train, priv_group=self.priv_group)
                        base_estimator.fit(model.transform(X_train), y_train)
                    else: pass # categorical feature conflict: LFR_pre requires num_var, TabTrans requires at least one cat_var
                    model = base_estimator # but now the base estimator is fit on lfr transformed training data
                elif BM=='LFR_in': model.fit(X_train,y_train,priv_group=self.priv_group)
                else: 
                    if BM in bm_category['IN']: 
                        model.fit(X_train, y_train)
                    elif base!='TabTrans': model.fit(X_train, y_train)
                    else: 
                        base_estimator.compile(optimizer = optimizer,
                                               loss = tf.keras.losses.BinaryCrossentropy(),
                                               metrics= [tf.keras.metrics.AUC(name="PR AUC", curve='PR')],)
                        model = base_estimator
                        model.fit(train_dataset, epochs=NUM_EPOCHS, 
                                      validation_data=val_dataset, callbacks=callback_list)
                # predict
                if base!='TabTrans': 
                    pred_prob_all = model.predict_proba(X_test)
                    pred_prob = pred_prob_all[:,1]
                else: 
                    pred_prob = model.predict(test_dataset)
                    pred_prob = pred_prob.reshape((pred_prob.shape[0],))
                    pred_prob_all = np.array([[1-pred, pred] for pred in pred_prob])

                for threshold in thres_arr:
                    y_pred = (pred_prob >= threshold).astype('int') # set threshold
                    # unique, counts = np.unique(y_pred, return_counts=True)
                    # print(np.asarray((unique, counts)).T)
                    
                    if len(set.intersection(set(BMs),set(bm_category['POST']))) == 0: # if none of the POST BM is included
                        thres_dict = store_metrics(y_test, y_pred, X_test, pred_prob, thres_dict, BM_name, threshold,
                                                   self.priv_group, self.pos_label, self.prot_attr)
                    else: # Post-Processing
                        if 'ROC' in BMs:
                            # fit the primary prediction to post-processing models 
                            print(y_test.index.to_frame().race.value_counts())
                            post_model = RejectOptionClassifier(prot_attr=self.prot_attr,threshold=threshold)
                            post_model.fit(pred_prob_all,y_test,pos_label=self.pos_label,priv_group=self.priv_group)
                        elif 'CEO' in BMs:
                            post_model = CalibratedEqualizedOdds(prot_attr=self.prot_attr,random_state=self.random_state)
                            post_model.fit(pred_prob_all,y_test,pos_label=self.pos_label)                            
                        # get final prediction
                        initial_pred = pd.DataFrame(pred_prob_all, index=X_test.index)
                        final_pred_prob = post_model.predict_proba(initial_pred)[:,1]
                        final_y_pred = (final_pred_prob >= threshold).astype('int') # set threshold
                        thres_dict = store_metrics(y_test, final_y_pred, X_test, final_pred_prob, thres_dict, BM_name,
                                                   threshold, self.priv_group, self.pos_label, self.prot_attr)
        return thres_dict
        
    # @methods
    def get_metrics(self, X, y, base='LR', param = {}, BM_arr=[None], thres_arr=[0.5]):
        # train classifier with stratified kfold cross validation
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.random_state)
        import copy
        all_metrics_dict = {metric:[] for metric in accuracy_metrics+fairness_metrics}
        thres_dict = {str(BM): {threshold: copy.deepcopy(all_metrics_dict) for threshold in thres_arr} for BM in BM_arr}
        thres_BM_dict = {}
        with mp.Pool(self.n_jobs) as pool:
            fixed_args = [X, y, base, param, BM_arr, thres_arr, thres_dict]
            index_args = skf.split(X, y)
            args = [(one,*two) for one, two in zip(index_args, repeat(fixed_args))]
            args = [(ele[0]) + tuple(ele[1:]) for ele in args]  
            print('-'*90)
            print('start multiprocessing')
            print('-'*90)
            thres_dict = pool.starmap(self.run_search, args)
            # thres_dict = merge_dictionary_list(thres_dict)
            pool.close()
            # pool.join()
        thres_dict = merge_dictionary_list(thres_dict)
        avg_metrics = get_avg_metrics(thres_dict)
        thres_BM_dict.update(avg_metrics)
        # print(thres_BM_dict)

        return thres_BM_dict
    

    
class fair_GridsearchCV():
    def __init__(self, base, param_grid, pos_label, priv_group, prot_attr='race', cv=10, random_state=1234, 
                 n_jobs=multiprocessing.cpu_count()-1):
        """
        base (string): base estimator, e.g. "LR", "RF", "GB", "SVM", "NB", "TabTr"
        """
        self.base = base
        self.param_grid = param_grid
        self.prot_attr = prot_attr
        self.pos_label = pos_label
        self.priv_group = priv_group
        self.cv = cv
        self.random_state = random_state
        self.n_jobs = n_jobs
        

    def fit(self, X, y,scoring=('norm_mcc','spd',1,1), random_state=1234):
        """Run fit with all sets of parameters.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples and
            `n_features` is the number of features.
        y : array-like of shape (n_samples, n_output) \
            or (n_samples,), default=None
            Target relative to X for classification or regression;
            None for unsupervised learning.
        scoring : tuple, default=('MCC','SPD',1,1)
                  1st element stands for the chosen accuracy metric;
                  2nd element stands for the chosen fairness metric;
                  3rd element stands for alpha value, that is, the weight on accuray cost;
                  4th element stands for beta value, that is, the weight on fairness cost;
                  
                  References:
                  Haas, Christian. "The price of fairness-A framework to explore trade-offs in algorithmic fairness." 
                  40th International Conference on Information Systems, ICIS 2019. Association for Information Systems, 2019.

        Returns
        -------
        self : object
            Instance of fitted estimator.
        """
        # first check model compatibilities
        if (self.base=='TabTrans') and (any(BM in self.param_grid['Bias_Mitigation'] for BM in ['LFR_pre','AD','EGR'])):
            print('********CONFLICT********')
            warnings.warn("TabTransformer conficts with LFR_pre, AD and EGR, please try other combinations.")
        else: 
            # default setting if hyper-parameters not specified
            default_hyperp_dict = {'LR': [{'penalty':'l2'}],
                                   'RF': [{'n_estimators':'100','criterion':'gini'}],
                                   'GB': [{'learning_rate':0.1,'criterion':'friedman_mse'}],
                                   'SVM':[{'C':1,'kernel':'rbf','gamma':'scale'}],
                                   'NB': [{'var_smoothing': 1e-9}],
                                   'TabTrans': [{'epochs': 10, 'learning_rate': 0.0001, 'weight_decay': 0.0001}]}
            if 'hyperp_grid' in self.param_grid: 
                hyperp_grid = list(ParameterGrid(self.param_grid['hyperp_grid']))
            else: hyperp_grid = default_hyperp_dict[self.base]  
            if 'Bias_Mitigation' not in self.param_grid: 
                self.param_grid['Bias_Mitigation'] = [None]        
            if 'threshold' not in self.param_grid: 
                self.param_grid['threshold'] = [0.5]

            all_metrics = {}
            for i, param in enumerate(tqdm(hyperp_grid)):
                print(param)
                model = skf_model(self.prot_attr, self.pos_label, self.priv_group, self.cv, self.random_state, self.n_jobs)
                metrics = model.get_metrics(X=X, y=y, base=self.base, param=param, 
                                            BM_arr=self.param_grid['Bias_Mitigation'], thres_arr=self.param_grid['threshold'])
                for j, BM in enumerate(metrics.keys()):
                    for k,thres in enumerate(metrics[BM].keys()):
                        all_param = {'hparam':param, 'Bias_Mitigation':BM, 'threshold':thres}
                        all_metrics['%s_%s%s%s'%(self.base,i,j,k)] = {'parameters':all_param, 'metrics':metrics[BM][thres]}

            self.all_metrics = all_metrics
            self.output_table = get_output_table(all_metrics, base=self.base, scoring=scoring)

            # "best" model
            param_col = ['base_estimator','param','Bias_Mitigation','threshold']
            self._best_index = np.argmin(self.output_table.cost)
            self._best_param = self.output_table.loc[self._best_index, param_col]
    #         try: 
    #             self.all_metrics = all_metrics
    #             self.output_table = get_output_table(all_metrics, base=self.base, scoring=scoring)

    #             # "best" model
    #             param_col = ['base_estimator','param','Bias_Mitigation','threshold']
    #             self._best_index = np.argmin(self.output_table.cost)
    #             self._best_param = self.output_table.loc[self._best_index, param_col]

    #         except: 
    #             warnings.warn("No Model was built, potential reasons include: \
    #                           1) TabTransformer conficts with LFR_pre regarding categorical features \
    #                           2) TabTransformer conficts with EGR because AIF360.sklearn only allows sklearn models.")

    #             # print('No Model was built, potential reasons include: \n 1) TabTransformer conficts with LFR_pre regarding categorical features \n2) TabTransformer conficts with EGR because AIF360.sklearn only allows sklearn models.')


        return self