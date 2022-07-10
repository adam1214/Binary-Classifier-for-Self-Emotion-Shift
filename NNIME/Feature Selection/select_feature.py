# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 20:39:29 2021

@author: Admin
"""
import joblib
import pandas as pd
import pdb
from collections import Counter
import numpy as np
import math
import matplotlib.pyplot as plt
import random
import os
import pickle
import pandas  as pd
from feature_selector import FeatureSelector
import sys
import copy
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats


def remove_high_correlation_feature_func(high_correlation_feature, remove_low_importance_train_feature):
    remove_low_importance_and_high_correlation_train_feature = copy.deepcopy(remove_low_importance_train_feature)
    for i in range(len(high_correlation_feature)):
        feature_name = high_correlation_feature[i]
        if(feature_name in remove_low_importance_and_high_correlation_train_feature):
            remove_low_importance_and_high_correlation_train_feature = remove_low_importance_and_high_correlation_train_feature.drop(columns=feature_name)
    
    return remove_low_importance_and_high_correlation_train_feature

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def cal_cosine_similarity(center_logits, target_logits, dim): # -1~1
    return cosine_similarity(center_logits.reshape(1, dim), target_logits.reshape(1, dim))[0][0]

def cal_kl_divergence(center_logits, target_logits): # 0~inf
    center_probs = softmax(center_logits)
    target_probs = softmax(target_logits)
    return scipy.stats.entropy(center_probs, target_probs) 

def cal_earth_mover_dist(center_logits, target_logits): # 0~inf
    return scipy.stats.wasserstein_distance(center_logits, target_logits)

def generate_interaction_sample(index_words, seq_dict, emo_dict, val=False):
    """ 
    Generate interaction training pairs,
    total 4 class, total 5531 emo samples."""
    emo = ['ang', 'hap', 'neu', 'sad']
    center_, target_, opposite_ = [], [], []
    center_label, target_label, opposite_label = [], [], []
    target_dist = []
    opposite_dist = []
    self_emo_shift = []
    for index, center in enumerate(index_words):
        if emo_dict[center] in emo or val == True:
            if emo_dict[center] in emo:
                four_type_utt_list.append(center)
            center_.append(center)
            center_label.append(emo_dict[center])
            pt = []
            pp = []
            for word in index_words[max(0, index - 8): index]:
                if word[-4] == center[-4]:
                    pt.append(word)
                else:
                    pp.append(word)

            if len(pt) != 0:
                target_.append(pt[-1])
                target_label.append(emo_dict[pt[-1]])
                target_dist.append(index - index_words.index(pt[-1]))
                if emo_dict[pt[-1]] == emo_dict[center]:
                    self_emo_shift.append(0)
                else:
                    self_emo_shift.append(1)
            else:
                target_.append('pad')
                target_label.append('pad')
                target_dist.append('None')
                self_emo_shift.append(0)

            if len(pp) != 0:
                opposite_.append(pp[-1])
                opposite_label.append(emo_dict[pp[-1]])
                opposite_dist.append(index - index_words.index(pp[-1]))
            else:
                opposite_.append('pad')
                opposite_label.append('pad')
                opposite_dist.append('None')

    return center_, target_, opposite_, center_label, target_label, opposite_label, target_dist, opposite_dist, self_emo_shift

def generate_interaction_data(dialog_dict, seq_dict, emo_dict, val_set, mode='context'):
    """Generate training/testing data (emo_train.csv & emo_test.csv) under specific modes.
    
    Args:
        mode:
            if mode == context: proposed transactional contexts, referred to IAAN.
            if mode == random: randomly sampled contexts, referred to baseline randIAAN.
    """
    center_train, target_train, opposite_train, center_label_train, target_label_train, opposite_label_train, target_dist_train, opposite_dist_train, self_emo_shift_train = [], [], [], [], [], [], [], [], []
    center_val, target_val, opposite_val, center_label_val, target_label_val, opposite_label_val, target_dist_val, opposite_dist_val, self_emo_shift_val = [], [], [], [], [], [], [], [], []
    if mode=='context':
        generator = generate_interaction_sample

    for k in dialog_dict.keys():
        dialog_order = dialog_dict[k]
        # training set
        if val_set not in k:
            c, t, o, cl, tl, ol, td, od, ses = generator(dialog_order, seq_dict, emo_dict)
            center_train += c
            target_train += t
            opposite_train += o
            center_label_train += cl
            target_label_train += tl
            opposite_label_train += ol
            target_dist_train += td
            opposite_dist_train += od
            self_emo_shift_train += ses
        # validation set
        else:
            c, t, o, cl, tl, ol, td, od, ses = generator(dialog_order, seq_dict, emo_dict, val=True)
            center_val += c
            target_val += t
            opposite_val += o
            center_label_val += cl
            target_label_val += tl
            opposite_label_val += ol
            target_dist_val += td
            opposite_dist_val += od
            self_emo_shift_val += ses

    # save dialog pairs to train.csv and test.csv
    train_filename= './emo_train.csv'
    val_filename= './emo_test.csv'
    column_order = ['center', 'target', 'opposite', 'center_label', 'target_label', 'opposite_label', 'target_dist', 'opposite_dist', 'self_emo_shift']
    # train
    d = {'center': center_train, 'target': target_train, 'opposite': opposite_train, 'center_label': center_label_train, 
         'target_label': target_label_train, 'opposite_label': opposite_label_train, 'target_dist': target_dist_train, 'opposite_dist': opposite_dist_train, 'self_emo_shift': self_emo_shift_train}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(train_filename, sep=',', index = False)
    # validation
    d = {'center': center_val, 'target': target_val, 'opposite': opposite_val, 'center_label': center_label_val, 
         'target_label': target_label_val, 'opposite_label': opposite_label_val, 'target_dist': target_dist_val, 'opposite_dist': opposite_dist_val, 'self_emo_shift': self_emo_shift_val}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(val_filename, sep=',', index = False)

sys.setrecursionlimit(2000)
    
if __name__ == "__main__":
    utt_logits_outputs_fold1 = joblib.load('../data/original_iaan/utt_logits_outputs_fold1.pkl')
    #utt_logits_outputs_fold2 = joblib.load('../data/original_iaan/utt_logits_outputs_fold2.pkl')
    #utt_logits_outputs_fold3 = joblib.load('../data/original_iaan/utt_logits_outputs_fold3.pkl')
    #utt_logits_outputs_fold4 = joblib.load('../data/original_iaan/utt_logits_outputs_fold4.pkl')
    #utt_logits_outputs_fold5 = joblib.load('../data/original_iaan/utt_logits_outputs_fold5.pkl')
    
    utt_hc_fold1 = joblib.load('../data/original_iaan/utt_hc_fold1.pkl')
    #utt_hc_fold2 = joblib.load('../data/original_iaan/utt_hc_fold2.pkl')
    #utt_hc_fold3 = joblib.load('../data/original_iaan/utt_hc_fold3.pkl')
    #utt_hc_fold4 = joblib.load('../data/original_iaan/utt_hc_fold4.pkl')
    #utt_hc_fold5 = joblib.load('../data/original_iaan/utt_hc_fold5.pkl')
    
    utt_hp_fold1 = joblib.load('../data/original_iaan/utt_hp_fold1.pkl')
    #utt_hp_fold2 = joblib.load('../data/original_iaan/utt_hp_fold2.pkl')
    #utt_hp_fold3 = joblib.load('../data/original_iaan/utt_hp_fold3.pkl')
    #utt_hp_fold4 = joblib.load('../data/original_iaan/utt_hp_fold4.pkl')
    #utt_hp_fold5 = joblib.load('../data/original_iaan/utt_hp_fold5.pkl')
    
    utt_hr_fold1 = joblib.load('../data/original_iaan/utt_hr_fold1.pkl')
    #utt_hr_fold2 = joblib.load('../data/original_iaan/utt_hr_fold2.pkl')
    #utt_hr_fold3 = joblib.load('../data/original_iaan/utt_hr_fold3.pkl')
    #utt_hr_fold4 = joblib.load('../data/original_iaan/utt_hr_fold4.pkl')
    #utt_hr_fold5 = joblib.load('../data/original_iaan/utt_hr_fold5.pkl')
    
    feat_pooled = joblib.load('../data/feat_preprocessing.pkl')
    # label
    emo_all_dict = joblib.load('../data/emo_all.pkl')
    emo_shift_all = joblib.load('../data/4emo_shift_all_rearrange.pkl')
    dialog_dict = joblib.load('../data/dialog_rearrange_4emo_iemocap.pkl')
    
    four_type_utt_list = []
    generate_interaction_data(dialog_dict, feat_pooled, emo_all_dict, val_set='Ses01')
    emo_train = pd.read_csv('./emo_train.csv')
    train_features, train_labels = [], []
    
    for index, row in emo_train.iterrows():
        train_labels.append(emo_shift_all[row['center']])
        
        hc = utt_hc_fold1[row['center']]
        if row['target'] == 'pad':
            hp = np.zeros(512)
        else:
            hp = utt_hp_fold1[row['target']]
        if row['opposite'] == 'pad':
            hr = np.zeros(512)
        else:
            hr = utt_hr_fold1[row['opposite']]
        
        hc_hp_cos_sim = cal_cosine_similarity(center_logits=hc, target_logits=hp, dim=512)
        hc_hp_kl_div = cal_kl_divergence(center_logits=hc, target_logits=hp)
        hc_hp_earth_mover_dist = cal_earth_mover_dist(center_logits=hc, target_logits=hp)
        
        hc_hr_cos_sim = cal_cosine_similarity(center_logits=hc, target_logits=hr, dim=512)
        hc_hr_kl_div = cal_kl_divergence(center_logits=hc, target_logits=hr)
        hc_hr_earth_mover_dist = cal_earth_mover_dist(center_logits=hc, target_logits=hr)
            
        Fc = feat_pooled[row['center']].flatten()
        Fp = feat_pooled[row['target']].flatten()
        Fr = feat_pooled[row['opposite']].flatten()
        
        Ec = utt_logits_outputs_fold1[row['center']]
        if row['target'] == 'pad':
            Ep = np.zeros(4)
            Ec_Ep_cos_sim = 1
            Ec_Ep_kl_div = 0
            Ec_Ep_earth_mover_dist = 0
        else:
            Ep = utt_logits_outputs_fold1[row['target']]
            Ec_Ep_cos_sim = cal_cosine_similarity(center_logits=Ec, target_logits=Ep, dim=4)
            Ec_Ep_kl_div = cal_kl_divergence(center_logits=Ec, target_logits=Ep)
            Ec_Ep_earth_mover_dist = cal_earth_mover_dist(center_logits=Ec, target_logits=Ep)
        if row['opposite'] == 'pad':
            Er = np.zeros(4)
            Ec_Er_cos_sim = 1
            Ec_Er_kl_div = 0
            Ec_Er_earth_mover_dist = 0
        else:
            Er = utt_logits_outputs_fold1[row['opposite']]
            Ec_Er_cos_sim = cal_cosine_similarity(center_logits=Ec, target_logits=Er, dim=4)
            Ec_Er_kl_div = cal_kl_divergence(center_logits=Ec, target_logits=Er)
            Ec_Er_earth_mover_dist = cal_earth_mover_dist(center_logits=Ec, target_logits=Er)
        # dim: 512*3+90*3+4*3+3 = 1821
        # train_features.append(np.concatenate([hc, hp, hr, Fc, Fp, Fr, Ec, Ep, Er, np.array([cos_sim, kl_div, earth_mover_dist])]))
        train_features.append(np.concatenate([np.array([Ec_Ep_cos_sim, Ec_Ep_kl_div, Ec_Ep_earth_mover_dist, Ec_Er_cos_sim, Ec_Er_kl_div, Ec_Er_earth_mover_dist, hc_hp_cos_sim, hc_hp_kl_div, hc_hp_earth_mover_dist, hc_hr_cos_sim, hc_hr_kl_div, hc_hr_earth_mover_dist])]))
    train_features = np.array(train_features)
    train_features_df = pd.DataFrame(train_features)
    col_names = ['Ec_Ep_cos_sim', 'Ec_Ep_kl_div', 'Ec_Ep_earth_mover_dist', 'Ec_Er_cos_sim', 'Ec_Er_kl_div', 'Ec_Er_earth_mover_dist', 'hc_hp_cos_sim', 'hc_hp_kl_div', 'hc_hp_earth_mover_dist', 'hc_hr_cos_sim', 'hc_hr_kl_div', 'hc_hr_earth_mover_dist']
    '''
    for i in range(0, 512, 1):
        col_names.append('hc_'+str(i))
    for i in range(0, 512, 1):
        col_names.append('hp_'+str(i))
    for i in range(0, 512, 1):
        col_names.append('hr_'+str(i))
    for i in range(0, 90, 1):
        col_names.append('Fc_'+str(i))
    for i in range(0, 90, 1):
        col_names.append('Fp_'+str(i))
    for i in range(0, 90, 1):
        col_names.append('Fr_'+str(i))
    for i in range(0, 4, 1):
        col_names.append('Ec_'+str(i))
    for i in range(0, 4, 1):
        col_names.append('Ep_'+str(i))
    for i in range(0, 4, 1):
        col_names.append('Er_'+str(i))
    
    col_names.append('cos_sim')
    col_names.append('kl_div')
    col_names.append('earth_mover_dist')
    '''
    train_features_df.columns = col_names
    
    # 创建 feature-selector 实例，并传入features 和labels
    fs = FeatureSelector(data=train_features_df, labels=np.array(train_labels, dtype=np.float32))
    
    # 选择zero importance的feature,
    # 
    # 参数说明：
    #          task: 'classification' / 'regression', 如果数据的模型是分类模型选择'classificaiton',
    #                否则选择'regression'
    #          eval_metric: 判断提前停止的metric. for example, 'auc' for classification, and 'l2' for regression problem
    #          n_iteration: 训练的次数
    #          early_stopping: True/False, 是否需要提前停止
    fs.identify_zero_importance(task='classification', eval_metric='auc', n_iterations=100, early_stopping=True)
    
    # 查看选择出的zero importance feature
    zero_importance_feature = fs.ops['zero_importance']
    
    # 绘制feature importance 关系图
    # 参数说明：
    #          plot_n: 指定绘制前plot_n个最重要的feature的归一化importance条形图，如图4所示
    #          threshold: 指定importance分数累积和的阈值，用于指定图4中的蓝色虚线.
    #              蓝色虚线指定了importance累积和达到threshold时，所需要的feature个数。
    #              注意：在计算importance累积和之前，对feature列表安装feature importance的大小
    #                   进行了降序排序
    # 
    #      
    fs.plot_feature_importances(threshold=0.9, plot_n=12)
    feature_importance = fs.feature_importances
    feature_importance.to_csv('feature_importance.csv', sep=',', index = False)
    
    # 选择出对importance累积和达到90%没有贡献的feature
    fs.identify_low_importance(cumulative_importance=0.9)
    
    # 查看选择出的feature
    low_importance_feature = fs.ops['low_importance']
    
    # 不对feature进行one-hot encoding（默认为False）, 然后选择出相关性大于95%的feature, 
    fs.identify_collinear(correlation_threshold=0.95, one_hot=False)
    
    # 查看选择的feature
    collinear_feature = fs.ops['collinear']
    collinear_feature_value = fs.record_collinear.head()
    # 绘制选择的特征的相关性heatmap
    #fs.plot_collinear(plot_all = False, fontsize = 10)
    #fs.plot_collinear(plot_all = True, fontsize = 1)
    
    # 绘制所有特征的相关性heatmap
    
    one_hot_features = fs.one_hot_features
    base_features = fs.base_features
    