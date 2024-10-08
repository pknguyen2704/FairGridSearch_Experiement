# helper file that collects functions used for analysis for the FairGridSearch results

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import re

def behaviour_analysis(data, metric_list, category='metric'):
    """
    Analyze how accuracy/fairness changes after applying bias mitigations.
    Args:
        data (dataframe): dataframe that stores all the results 
        metric_list (list): list of metrics to anaylze on, e.g. accuracy/fairness metrics
        category (str): if not "metric", the analysis is run w.r.t. this category, e.g. "BM","base"
    Function that runs the analysis Chen et al. included in their work (fig. 3-6)
    ref: Chen, Z., Zhang, J. M., Sarro, F., and Harman, M. (2023). 
         "A Comprehensive Empirical Study of Bias Mitigation Methods for Machine Learning Classifiers."
         (https://github.com/chenzhenpeng18/TOSEM23-BiasMitigationStudy/tree/main/Analysis_code)
    """
    from numpy import mean, std, sqrt
    import scipy.stats as stats

    def cohen_d(x, y):
        return abs(mean(x) - mean(y)) / sqrt((std(x, ddof=1) ** 2 + std(y, ddof=1) ** 2) / 2.0)

    def mann(x, y):
        return stats.mannwhitneyu(x, y)[1]
    
    the_category = category
    # construct diff_degree structure based on the category
    diff_degree = {}
    if the_category == 'metric':
        for metric in metric_list:
            diff_degree[metric] = {}
            for scale in ['noorincrease', 'small', 'medium', 'large']:
                diff_degree[metric][scale] = 0
    elif the_category == 'bm':     
        for BM in data.Bias_Mitigation.unique():
            if BM=='None': pass
            else:
                diff_degree[BM] = {}
                for scale in ['noorincrease', 'small', 'medium', 'large']:
                    diff_degree[BM][scale] = 0
    elif the_category == 'base':
        for base in data.base_estimator.unique():
            diff_degree[base] = {}
            for scale in ['noorincrease', 'small', 'medium', 'large']:
                diff_degree[base][scale] = 0
                
    # analyze changes
    if 'dataset' not in data.columns:
        data.insert(0,'dataset','dataset') # work around for analysing all datasets combined 
    for dataset in data.dataset.unique():
            for metric in metric_list:
                for base in data.base_estimator.unique():
                    default = data[(data.dataset==dataset)&\
                                        (data.base_estimator==base)&\
                                        (data.Bias_Mitigation=='None')]
                    default = default.sort_values(by=['threshold','param']).reset_index(drop=True)
                    default_list = default[metric]
                    default_mean = default_list.mean()
                    for BM in data[data.base_estimator==base].Bias_Mitigation.unique():
                        if BM == 'None': pass
                        else: 
                            bm = data[(data.dataset==dataset)&\
                                           (data.base_estimator==base)&\
                                           (data.Bias_Mitigation==BM)]
                            bm = bm.sort_values(by=['threshold','param']).reset_index(drop=True)
                            bm_list = bm[metric]
                            if (default.iloc[:, [0,1,2,4]] != bm.iloc[:, [0,1,2,4]]).any().any():
                                if (BM in ['AD','LFR_in'])&((default.iloc[:, [0,1,4]]==bm.iloc[:, [0,1,4]]).all().all()): pass
                                else: 
                                    print(dataset, base, BM)
                            bm_mean = bm_list.mean()
                            rise_ratio = bm_mean - default_mean
                            cohenn = cohen_d(default_list, bm_list)
                            # store values according to category                    
                            category = metric if the_category == 'metric' else (BM if the_category == 'bm' else base)
                            if mann(default_list, bm_list) >= 0.05 or rise_ratio >= 0:
                                diff_degree[category]['noorincrease'] += 1
                            elif cohenn < 0.5:
                                diff_degree[category]['small'] += 1
                            elif cohenn >= 0.5 and cohenn < 0.8:
                                diff_degree[category]['medium'] += 1
                            elif cohenn >= 0.8:
                                diff_degree[category]['large'] += 1
    # table = pd.DataFrame(diff_degree)
    # table = table.apply(lambda x: x/x.sum())
    # table.columns = [col.removeprefix('abs_').removeprefix('avg_').removesuffix('_score').upper() for col in table.columns]
    # # shorten "1-consistency" name
    # table = table.rename({'(1-CONSISTENCY_SCORE)': '1-CNS'}, axis=1)
    # # re-order columns
    # if 'LR' in table.columns:
    #     table = table[['LR','RF','GB','SVM','NB','TABTRANS']]
    # elif 'RW' in table.columns:
    #     table = table[['RW','LFR_PRE','LFR_IN','AD','EGR','ROC','CEO','RW+ROC','RW+CEO']]

    return diff_degree    

def plot_behaviour_analysis(diff_degree, data_name, category='metrics', caption='', figsize=(8, 6)):
    # plot tutorial: https://sharkcoder.com/data-visualization/mpl-stacked-bars
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import seaborn as sns
    sns.set()
    import re
    
    if data_name != 'All':    # construct table for single datasets
        table = pd.DataFrame(diff_degree)
        # --------------------- save original table ---------------------
        filename = caption.split()[0]+'_'+category
        table.to_pickle('./Result_Diff_Degree/'+data_name+'_'+filename)
        # --------------------------------------------------------------- 
    else: table = diff_degree.copy()
    
    table = table.apply(lambda x: x/x.sum())
    table.columns = [col.removeprefix('abs_').removeprefix('avg_').removesuffix('_score').upper() for col in table.columns]
    # shorten "1-consistency" name
    table = table.rename({'(1-CONSISTENCY_SCORE)': '1-CNS'}, axis=1)
    # re-order columns
    if 'LR' in table.columns:
        table = table[['LR','RF','GB','SVM','NB','TABTRANS']]
    elif 'RW' in table.columns:
        table = table[['RW','LFR_PRE','LFR_IN','AD','EGR','ROC','CEO','RW+ROC','RW+CEO']]

    # table.index = ['Increase or no significance', 'Decrease (small)', 'Decrease (medium)', 'Decrease (large)']
    table.index = ['N or (+)',  '(-) Small', '(-) Medium', '(-) Large']

    font_color = '#525252'
    # csfont = {'fontname':'Georgia'} # title font
    # hfont = {'fontname':'Calibri'} # main font
    # colors = ['#f47e7a', '#b71f5c', '#621237', '#dbbaa7']
    colors = ['#F8F8F8','#DCDCDC','#A9A9A9','#696969']

    # 1. Create the plot
    data = table.copy().T
    ax = data.plot.bar(align='center', stacked=True, figsize=figsize, color=colors, width=0.3, edgecolor="none")
    plt.tight_layout()

    # 2. Create the title        
    title = plt.title(caption.format(data_name.replace('_',' ')), pad=60, fontsize=11, color=font_color, y=0.5)
    title.set_position([.5, 0.95])
    # Adjust the subplot so that the title would fit
    plt.subplots_adjust(top=0.8, left=0.26)

    # 3. Set labels’ and ticks’ font size and color
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontsize(10)
    plt.xticks(rotation=0, color=font_color)
    plt.yticks(color=font_color)
    plt.ylabel('Proportions of Scenarios')

    # 4. Add legend    
    legend = plt.legend(
           loc='center',
           handletextpad=0.3,
           handlelength=1.2,
           handleheight=1.2,
           frameon=False,
           bbox_to_anchor=(0., 1.02, 1., .102), 
           mode='expand', 
           ncol=4, 
           borderaxespad=-.46,
           prop={'size': 15})

    for text in legend.get_texts():
        plt.setp(text, color=font_color) # legend font color
    # 5. Add annotations      
    for p in ax.patches:
        width, height = p.get_width(), p.get_height()
        x, y = p.get_xy() 
        if height>0:
            ax.text(x+width/2, 
                    y+height/2, 
                    '{0:.2f}%'.format(height*100), 
                    horizontalalignment='center', 
                    verticalalignment='center',
                    color='black',
                    # weight='bold',
                    fontsize=8)
    # save plot
    filename = caption.split()[0]+'_'+category
    plt.savefig('./Result_Plots/'+data_name+'_'+filename+'.png', bbox_inches='tight')
    

def get_df_metric_diff(data, metric_list):
    if 'dataset' not in data.columns:
        data.insert(0,'dataset','dataset') # work around for analysing all datasets combined 

    df_metric_diff = pd.DataFrame()
    for dataset in data.dataset.unique():
        df_base_BM = pd.DataFrame(columns=['dataset','base_estimator','Bias_Mitigation'])
        for base in data.base_estimator.unique():
            df_base = pd.DataFrame(columns=['dataset','base_estimator','Bias_Mitigation'])
            for metric in metric_list:
                default = data[(data.dataset==dataset)&\
                                    (data.base_estimator==base)&\
                                    (data.Bias_Mitigation=='None')]
                default = default.sort_values(by=['threshold','param']).reset_index(drop=True)
                default_list = default[metric]
                df_base_BM = pd.DataFrame()
                for BM in data[data.base_estimator==base].Bias_Mitigation.unique():
                    if BM == 'None': pass
                    else: 
                        bm = data[(data.dataset==dataset)&\
                                       (data.base_estimator==base)&\
                                       (data.Bias_Mitigation==BM)]
                        bm = bm.sort_values(by=['threshold','param']).reset_index(drop=True)
                        bm_list = bm[metric]
                        # print(bm_list)
                        if (default.iloc[:, [0,1,2,4]] != bm.iloc[:, [0,1,2,4]]).any().any():
                            if (BM in ['AD','LFR_in'])&((default.iloc[:, [0,1,4]]==bm.iloc[:, [0,1,4]]).all().all()): pass
                            else: 
                                print(dataset, base, BM)
                        change = bm_list-default_list
                        # print(dataset, base, BM, metric, change)
                        subset = pd.DataFrame(data={'dataset': dataset, 'base_estimator': base,
                                                    'Bias_Mitigation': BM}, index=range(len(change)))

                        subset[metric] = change
                        df_base_BM = pd.concat([df_base_BM, subset])
                        # display(subset)
                # display(df_base_BM)
                if df_base_BM.columns[-1]=='ACC': 
                    df_base = df_base_BM.copy()
                    standard_order = df_base_BM.iloc[:,:-1]
                else: 
                    if (df_base_BM.iloc[:,:-1].values==standard_order.values).all(): 
                        df_base[metric] = df_base_BM[metric]
                    else: print('Wrong order, cannot concat')

        df_metric_diff = pd.concat([df_metric_diff, df_base])
    return df_metric_diff

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def corr_heatmap_with_significance(df, annot=True, sig_levels=(0.05, 0.01, 0.001), acc=False, acc_fair=False):
    accuracy_map = ['ACC', 'BACC', 'F1', 'AUC', 'MCC', 'NORM_MCC']
    fairness_map = ['SPD', 'AOD', 'EOD', 'FORD', 'PPVD', '1-CNS', 'GEI', 'TI']
    space = 0 if acc else (0.1 if acc_fair else 0.05)
    corr_matrix = df.corr('spearman')
    if acc_fair:
        corr_matrix = df[accuracy_map+fairness_map].corr('spearman')
        corr_matrix.loc[accuracy_map, fairness_map]
    mask = np.zeros_like(corr_matrix, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("dark"):
        ax = sns.heatmap(corr_matrix, mask=mask, annot=annot, annot_kws={'fontsize': 10},
                         cmap='vlag', vmin=-1, vmax=1, fmt=".2f", center=0, square=True)
        for i in range(corr_matrix.shape[0]):
            for j in range(i+1, corr_matrix.shape[1]):
                p = stats.spearmanr(df.iloc[:, i], df.iloc[:, j])[1]
                sig_stars = ''
                for idx, level in enumerate(sig_levels):
                    if p <= level:
                        sig_stars += '*'
                if sig_stars:
                    ax.text(i+0.5, j+0.78+space, '{}'.format(sig_stars), ha='center', va='center', fontsize=12)
                    # ax.text(j+0.5, i+0.9+space, '{}'.format(sig_stars), ha='center', va='center', fontsize=12)
                else: pass
        if acc: ax.set_xticklabels(ax.get_xticklabels(), horizontalalignment='right')
        else: ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
    return ax

def corr_heatmap_with_significance_acc_fair(df, annot=True, sig_levels=(0.05, 0.01, 0.001)):
    accuracy_map = ['ACC', 'BACC', 'F1', 'AUC', 'MCC', 'NORM_MCC']
    fairness_map = ['SPD', 'AOD', 'EOD', 'FORD', 'PPVD', '1-CNS', 'GEI', 'TI']
    corr_matrix = df.corr('spearman')
    corr_matrix = corr_matrix.loc[fairness_map, accuracy_map]
    with sns.axes_style("dark"):
        ax = sns.heatmap(corr_matrix, annot=annot, annot_kws={'fontsize': 10},
                         cmap='vlag', vmin=-1, vmax=1, fmt=".2f", center=0, square=True)
        for i in range(corr_matrix.shape[0]):
            for j in range(corr_matrix.shape[1]):
                df_pvalue = pd.DataFrame(stats.spearmanr(df[accuracy_map+fairness_map]).pvalue,
                                         columns=accuracy_map+fairness_map, index=accuracy_map+fairness_map, )\
                                        .loc[fairness_map,accuracy_map]
                p = df_pvalue.iloc[i,j]
                # print(p)
                sig_stars = ''
                for idx, level in enumerate(sig_levels):
                    if p <= level:
                        sig_stars += '*'
                if sig_stars:
                    ax.text(j+0.5, i+0.9, '{}'.format(sig_stars), ha='center', va='center', fontsize=12)
                    # ax.text(j+0.5, i+0.9+space, '{}'.format(sig_stars), ha='center', va='center', fontsize=12)
                else: pass
        else: ax.set_yticklabels(ax.get_yticklabels(), rotation=0, horizontalalignment='right')
        return ax
    