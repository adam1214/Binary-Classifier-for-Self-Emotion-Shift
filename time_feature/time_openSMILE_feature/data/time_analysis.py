import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import pandas as pd
import pdb
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler 
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek, SMOTEENN
import seaborn as sns
import matplotlib.pyplot as plt

def generate_interaction_sample(index_words, seq_dict, emo_dict):
    global negative_sec_cnt
    """ 
    Generate interaction training pairs,
    total 4 class, total 5531 emo samples."""
    emo = ['ang', 'hap', 'neu', 'sad']
    center_, target_, opposite_ = [], [], []
    center_label, target_label, opposite_label = [], [], []
    target_dist = []
    opposite_dist = []
    self_emo_shift = []
    self_time_dur = []
    closest_time_dur = []
    for index, center in enumerate(index_words):
        if emo_dict[center] in emo:
            time_self_self = 10000.
            time_self_opp = 10000.
            #if True:
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
                time_self_self = utt_time_dict[center]['start_sec'] - utt_time_dict[pt[-1]]['start_sec']
                self_time_dur.append(time_self_self)
            else:
                target_.append('pad')
                target_label.append('pad')
                target_dist.append('None')
                self_emo_shift.append(0)
                self_time_dur.append('None')

            if len(pp) != 0:
                opposite_.append(pp[-1])
                opposite_label.append(emo_dict[pp[-1]])
                opposite_dist.append(index - index_words.index(pp[-1]))
                time_self_opp = utt_time_dict[center]['start_sec'] - utt_time_dict[pp[-1]]['start_sec']
                if time_self_opp < 0:
                    negative_sec_cnt += 1
                    print('negative sample', negative_sec_cnt, center, pp[-1])
            else:
                opposite_.append('pad')
                opposite_label.append('pad')
                opposite_dist.append('None')
            
            if min(time_self_opp, time_self_self) != 10000.:
                closest_time_dur.append(min(time_self_opp, time_self_self))
            else:
                closest_time_dur.append('None')

    return center_, target_, opposite_, center_label, target_label, opposite_label, target_dist, opposite_dist, self_emo_shift, self_time_dur, closest_time_dur

def generate_interaction_data(dialog_dict, seq_dict, emo_dict, mode='context'):
    """Generate training/testing data (emo_train.csv & emo_test.csv) under specific modes.
    
    Args:
        mode:
            if mode == context: proposed transactional contexts, referred to IAAN.
            if mode == random: randomly sampled contexts, referred to baseline randIAAN.
    """
    center_train, target_train, opposite_train, center_label_train, target_label_train, opposite_label_train, target_dist_train, opposite_dist_train, self_emo_shift_train, self_time_dur_train, closest_time_dur_train = [], [], [], [], [], [], [], [], [], [], []
    
    if mode=='context':
        generator = generate_interaction_sample

    for k in dialog_dict.keys():
        dialog_order = dialog_dict[k]
        c, t, o, cl, tl, ol, td, od, ses, std, ctd = generator(dialog_order, seq_dict, emo_dict)
        center_train += c
        target_train += t
        opposite_train += o
        center_label_train += cl
        target_label_train += tl
        opposite_label_train += ol
        target_dist_train += td
        opposite_dist_train += od
        self_emo_shift_train += ses
        self_time_dur_train += std
        closest_time_dur_train += ctd

    # save dialog pairs to train.csv and test.csv
    data_filename= './all_data.csv'
    column_order = ['center', 'target', 'opposite', 'center_label', 'target_label', 'opposite_label', 'target_dist', 'opposite_dist', 'self_emo_shift', 'self_time_dur', 'closest_time_dur']
    # train
    d = {'center': center_train, 'target': target_train, 'opposite': opposite_train, 'center_label': center_label_train, 
         'target_label': target_label_train, 'opposite_label': opposite_label_train, 'target_dist': target_dist_train, 'opposite_dist': opposite_dist_train, 'self_emo_shift': self_emo_shift_train, 'self_time_dur': self_time_dur_train, 'closest_time_dur': closest_time_dur_train}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(data_filename, sep=',', index = False)
        
def upsampling(X, Y):
    #counter = Counter(Y)
    #print(counter)
    
    # transform the dataset
    #oversample = SMOTE(random_state=100, n_jobs=-1, sampling_strategy='auto', k_neighbors=5)
    oversample = RandomOverSampler(random_state=100)
    #oversample = ClusterCentroids(random_state=100, n_jobs=-1)
    #oversample = SMOTETomek(random_state=100, n_jobs=-1, sampling_strategy='auto')
    X_upsample, Y_upsample = oversample.fit_resample(np.array(X).squeeze(1), Y)
    
    #counter = Counter(Y_upsample)
    #print(counter)

    return X_upsample, Y_upsample

if __name__ == "__main__":    
    negative_sec_cnt = 0
    utt_time_dict = joblib.load('./utt_time_dict.pkl')
    # dimension of each utterance: (n, 45)
    # n:number of time frames in the utterance
    emo_num_dict = {'ang': 0, 'hap': 1, 'neu':2, 'sad': 3, 'sur': 4, 'fru': 5, 'xxx': 6, 'oth': 7, 'fea': 8, 'dis': 9, 'pad': 10}
    feat_pooled = joblib.load('./feat_pooled.pkl')
    
    # label
    emo_all_dict = joblib.load('./emo_all.pkl')
    
    # dialog order
    dialog_dict = joblib.load('./dialog.pkl')

    # generate data
    generate_interaction_data(dialog_dict, feat_pooled, emo_all_dict)
    all_data = pd.read_csv('./all_data.csv')
    
    self_emo_shift_self_time_dur = []
    self_emo_shift_closest_time_dur = []
    self_emo_no_shift_self_time_dur = []
    self_emo_no_shift_closest_time_dur = []
    
    for i in range(len(all_data)):
        if all_data['self_emo_shift'][i] == 1 and all_data['self_time_dur'][i] != 'None':
            self_emo_shift_self_time_dur.append(float(all_data['self_time_dur'][i]))
        elif all_data['self_emo_shift'][i] == 0 and all_data['self_time_dur'][i] != 'None':
            self_emo_no_shift_self_time_dur.append(float(all_data['self_time_dur'][i]))
        
        if all_data['self_emo_shift'][i] == 1 and all_data['closest_time_dur'][i] != 'None':
            self_emo_shift_closest_time_dur.append(float(all_data['closest_time_dur'][i]))
        elif all_data['self_emo_shift'][i] == 0 and all_data['closest_time_dur'][i] != 'None':
            self_emo_no_shift_closest_time_dur.append(float(all_data['closest_time_dur'][i]))
    
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    plot = sns.distplot(pd.DataFrame(self_emo_shift_self_time_dur, columns=['self emo shift_self time dur (sec)'])['self emo shift_self time dur (sec)'], bins = int((max(self_emo_shift_self_time_dur)-min(self_emo_shift_self_time_dur))/1), norm_hist=False, ax=ax[0][0])
    plot = sns.distplot(pd.DataFrame(self_emo_shift_closest_time_dur, columns=['self emo shift_closest time dur (sec)'])['self emo shift_closest time dur (sec)'], bins = int((max(self_emo_shift_closest_time_dur)-min(self_emo_shift_closest_time_dur))/1), norm_hist=False, ax=ax[0][1])
    plot = sns.distplot(pd.DataFrame(self_emo_no_shift_self_time_dur, columns=['self emo no shift_self time dur (sec)'])['self emo no shift_self time dur (sec)'], bins = int((max(self_emo_no_shift_self_time_dur)-min(self_emo_no_shift_self_time_dur))/1), norm_hist=False, ax=ax[1][0])
    plot = sns.distplot(pd.DataFrame(self_emo_no_shift_closest_time_dur, columns=['self emo no shift_closest time dur (sec)'])['self emo no shift_closest time dur (sec)'], bins = int((max(self_emo_no_shift_closest_time_dur)-min(self_emo_no_shift_closest_time_dur))/1), norm_hist=False, ax=ax[1][1])
    plt.tight_layout()
    plt.savefig('density.png')

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))
    plot = sns.histplot(pd.DataFrame(self_emo_shift_self_time_dur, columns=['self emo shift_self time dur (sec)'])['self emo shift_self time dur (sec)'], bins = int((max(self_emo_shift_self_time_dur)-min(self_emo_shift_self_time_dur))/1), ax=ax[0][0])
    plot = sns.histplot(pd.DataFrame(self_emo_shift_closest_time_dur, columns=['self emo shift_closest time dur (sec)'])['self emo shift_closest time dur (sec)'], bins = int((max(self_emo_shift_closest_time_dur)-min(self_emo_shift_closest_time_dur))/1), ax=ax[0][1])
    plot = sns.histplot(pd.DataFrame(self_emo_no_shift_self_time_dur, columns=['self emo no shift_self time dur (sec)'])['self emo no shift_self time dur (sec)'], bins = int((max(self_emo_no_shift_self_time_dur)-min(self_emo_no_shift_self_time_dur))/1), ax=ax[1][0])
    plot = sns.histplot(pd.DataFrame(self_emo_no_shift_closest_time_dur, columns=['self emo no shift_closest time dur (sec)'])['self emo no shift_closest time dur (sec)'], bins = int((max(self_emo_no_shift_closest_time_dur)-min(self_emo_no_shift_closest_time_dur))/1), ax=ax[1][1])
    plt.tight_layout()
    plt.savefig('count.png')
    