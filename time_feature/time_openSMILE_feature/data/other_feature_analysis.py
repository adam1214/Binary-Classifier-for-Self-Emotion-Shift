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
from scipy.stats import norm
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats

def softmax(x):
    f_x = np.exp(x) / np.sum(np.exp(x))
    return f_x

def cal_cosine_similarity(center_logits, target_logits): # 0~1
    return cosine_similarity(center_logits.reshape(1, 4), target_logits.reshape(1, 4))[0][0]

def cal_kl_divergence(center_logits, target_logits): # 0~inf
    center_probs = softmax(center_logits)
    target_probs = softmax(target_logits)
    return scipy.stats.entropy(center_probs, target_probs) 

def cal_earth_mover_dist(center_logits, target_logits): # 0~inf
    return scipy.stats.wasserstein_distance(center_logits, target_logits)

def generate_interaction_sample(index_words, seq_dict, emo_dict, case_num):
    global negative_sec_cnt, closest_self_opp_cnt, total_closest
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
    cosine_similarity = []
    kl_divergence = []
    earth_mover_dist = []

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
                
                cosine_similarity.append(cal_cosine_similarity(center_logits = emo_outputs[center], target_logits = emo_outputs[pt[-1]]))
                kl_divergence.append(cal_kl_divergence(center_logits = emo_outputs[center], target_logits = emo_outputs[pt[-1]]))
                earth_mover_dist.append(cal_earth_mover_dist(center_logits = emo_outputs[center], target_logits = emo_outputs[pt[-1]]))
                
            else:
                target_.append('pad')
                target_label.append('pad')
                target_dist.append('None')
                self_emo_shift.append(0)
                self_time_dur.append('None')
                cosine_similarity.append('None')
                kl_divergence.append('None')
                earth_mover_dist.append('None')

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
                total_closest += 1
                if min(time_self_opp, time_self_self) == time_self_opp:
                    closest_self_opp_cnt += 1
                closest_time_dur.append(min(time_self_opp, time_self_self))
            else:
                closest_time_dur.append('None')
                
            if case_num == 1 and self_emo_shift[-1] == 1:
                del center_[-1]
                del target_[-1]
                del opposite_[-1]
                del center_label[-1]
                del target_label[-1]
                del opposite_label[-1]
                del target_dist[-1]
                del opposite_dist[-1]
                del self_emo_shift[-1]
                del self_time_dur[-1]
                del closest_time_dur[-1]
                del cosine_similarity[-1]
                del kl_divergence[-1]
                del earth_mover_dist[-1]
            elif case_num == 2 and self_emo_shift[-1] == 0:
                del center_[-1]
                del target_[-1]
                del opposite_[-1]
                del center_label[-1]
                del target_label[-1]
                del opposite_label[-1]
                del target_dist[-1]
                del opposite_dist[-1]
                del self_emo_shift[-1]
                del self_time_dur[-1]
                del closest_time_dur[-1]
                del cosine_similarity[-1]
                del kl_divergence[-1]
                del earth_mover_dist[-1]

    return center_, target_, opposite_, center_label, target_label, opposite_label, target_dist, opposite_dist, self_emo_shift, self_time_dur, closest_time_dur, cosine_similarity, kl_divergence, earth_mover_dist

def generate_interaction_data(dialog_dict, seq_dict, emo_dict, case_num, mode='context'):
    """Generate training/testing data (emo_train.csv & emo_test.csv) under specific modes.
    
    Args:
        mode:
            if mode == context: proposed transactional contexts, referred to IAAN.
            if mode == random: randomly sampled contexts, referred to baseline randIAAN.
    """
    center_train, target_train, opposite_train, center_label_train, target_label_train, opposite_label_train, target_dist_train, opposite_dist_train, self_emo_shift_train, self_time_dur_train, closest_time_dur_train = [], [], [], [], [], [], [], [], [], [], []
    cosine_similarity_train, kl_divergence_train, earth_mover_dist_train = [], [], []
    if mode=='context':
        generator = generate_interaction_sample

    for k in dialog_dict.keys():
        dialog_order = dialog_dict[k]
        c, t, o, cl, tl, ol, td, od, ses, std, ctd, cs, kld, emd = generator(dialog_order, seq_dict, emo_dict, case_num=case_num)
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
        cosine_similarity_train += cs
        kl_divergence_train += kld
        earth_mover_dist_train += emd
        
    # save dialog pairs to train.csv and test.csv
    
    if case_num != None:
        data_filename = './case' + str(case_num) + '_data.csv'
    else:
        data_filename = './all_data.csv'
    column_order = ['center', 'target', 'opposite', 'center_label', 'target_label', 'opposite_label', 'target_dist', 'opposite_dist', 'self_emo_shift', 'self_time_dur', 'closest_time_dur', 'cosine_similarity', 'kl_divergence', 'earth_mover_dist']
    # train
    d = {'center': center_train, 'target': target_train, 'opposite': opposite_train, 'center_label': center_label_train, 
         'target_label': target_label_train, 'opposite_label': opposite_label_train, 'target_dist': target_dist_train, 'opposite_dist': opposite_dist_train, 'self_emo_shift': self_emo_shift_train, 'self_time_dur': self_time_dur_train, 'closest_time_dur': closest_time_dur_train,
         'cosine_similarity': cosine_similarity_train, 'kl_divergence': kl_divergence_train, 'earth_mover_dist': earth_mover_dist_train}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(data_filename, sep=',', index = False)

if __name__ == "__main__":    
    case_1 = 0
    case_2 = 0
    negative_sec_cnt = 0
    closest_self_opp_cnt = 0
    total_closest = 0
    utt_time_dict = joblib.load('./utt_time_dict.pkl')
    # dimension of each utterance: (n, 45)
    # n:number of time frames in the utterance
    emo_num_dict = {'ang': 0, 'hap': 1, 'neu':2, 'sad': 3, 'sur': 4, 'fru': 5, 'xxx': 6, 'oth': 7, 'fea': 8, 'dis': 9, 'pad': 10}
    feat_pooled = joblib.load('./feat_pooled.pkl')
    
    # label
    emo_all_dict = joblib.load('./emo_all.pkl')
    emo_shift_all_dict = joblib.load('./4emo_shift_all_rearrange.pkl')
    
    #emo_outputs = joblib.load('./dag_outputs_4_all_fold_single_rearrange.pkl')
    emo_outputs = joblib.load('./rearrange_single_outputs_iaan.pkl')
    
    # check pretrained model performance
    preds, gts = [], []
    for utt in emo_all_dict:
        if emo_all_dict[utt] in ['ang', 'hap', 'neu', 'sad']:
            gts.append(emo_num_dict[emo_all_dict[utt]])
            preds.append(emo_outputs[utt].argmax())
    print('pretrained model UAR =', round(100*recall_score(gts, preds, average='macro'), 2), '%')
    print('pretrained model ACC =', round(100*accuracy_score(gts, preds), 2), '%')
    
    # dialog order
    dialog_dict = joblib.load('./dialog_rearrange_4emo_iemocap.pkl')
    #dialog_dict = joblib.load('./dialog_rearrange.pkl')
    
    # generate data
    for case_num in range(1, 3, 1):
        generate_interaction_data(dialog_dict, feat_pooled, emo_all_dict, case_num)
        data_filename = './case' + str(case_num) + '_data.csv'
        
        #all_data = pd.read_csv('./all_data.csv')
        all_data = pd.read_csv(data_filename)
        if case_num == 1:
            case_1 = len(all_data)
            print('########## EMO NO SHIFT##########')
        elif case_num == 2:
            case_2 = len(all_data)
            print('########## EMO SHIFT##########')

        cosine_similarity_list, kl_divergence_list, earth_mover_dist_list = [], [], []
        for i in range(len(all_data)):
            if all_data['cosine_similarity'][i] != 'None':
                cosine_similarity_list.append(float(all_data['cosine_similarity'][i]))
            if all_data['kl_divergence'][i] != 'None':
                kl_divergence_list.append(float(all_data['kl_divergence'][i]))
            if all_data['earth_mover_dist'][i] != 'None':
                earth_mover_dist_list.append(float(all_data['earth_mover_dist'][i]))
        print('cosine_similarity mean:', round(np.mean(cosine_similarity_list), 2))
        print('cosine_similarity std:', round(np.std(cosine_similarity_list), 2))
        
        print('kl_divergence mean:', round(np.mean(kl_divergence_list), 2))
        print('kl_divergence std:', round(np.std(kl_divergence_list), 2))
        
        print('earth_mover_dist mean:', round(np.mean(earth_mover_dist_list) ,2))
        print('earth_mover_dist std:', round(np.std(earth_mover_dist_list), 2))
    print('####################')
    print('case1:', round(100*case_1/(case_1+case_2), 2), '%')
    print('case2:', round(100*case_2/(case_1+case_2), 2), '%')
    
    '''
    val = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
    for val_ in val:
        print("################{}################".format(val_))
        for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            print('##########t =', t, '##########')
            emo_shift_preds = []
            emo_shift_gts = []
            generate_interaction_data(dialog_dict, feat_pooled, emo_all_dict, case_num=None)
            all_data = pd.read_csv('./all_data.csv')
            for i in range(len(all_data)):
                if all_data['cosine_similarity'][i] != 'None':
                    if float(all_data['cosine_similarity'][i]) >= t:
                        emo_shift_preds.append(0.0)
                    else:
                        emo_shift_preds.append(1.0)
                else:
                    emo_shift_preds.append(0.0)
                emo_shift_gts.append(emo_shift_all_dict[all_data['center'][i]])
            print('UAR:', round(recall_score(emo_shift_gts, emo_shift_preds, average='macro')*100, 2), '%')
            print('precision (predcit label 1):', round(precision_score(emo_shift_gts, emo_shift_preds)*100, 2), '%')
    '''
    '''
    # get emo_shift prob (0 or 1) from pretrained emo model
    emo_shift_preds = []
    emo_shift_gts = []
    generate_interaction_data(dialog_dict, feat_pooled, emo_all_dict, case_num=None)
    all_data = pd.read_csv('./all_data.csv')
    for i in range(len(all_data)):
        
        if all_data['target'][i] == 'pad' or emo_outputs[all_data['center'][i]].argmax() == emo_outputs[all_data['target'][i]].argmax():
            emo_shift_preds.append(0.0)
        else:
            emo_shift_preds.append(1.0)
        emo_shift_gts.append(emo_shift_all_dict[all_data['center'][i]])
    print('UAR:', round(recall_score(emo_shift_gts, emo_shift_preds, average='macro')*100, 2), '%')
    print('precision (predcit label 1):', round(precision_score(emo_shift_gts, emo_shift_preds)*100, 2), '%')
    '''