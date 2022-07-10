import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import pdb
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler 
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek, SMOTEENN
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
    cosine_similarity = []
    kl_divergence = []
    earth_mover_dist = []
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
                cosine_similarity.append(cal_cosine_similarity(center_logits = emo_outputs[center], target_logits = emo_outputs[pt[-1]]))
                kl_divergence.append(cal_kl_divergence(center_logits = emo_outputs[center], target_logits = emo_outputs[pt[-1]]))
                earth_mover_dist.append(cal_earth_mover_dist(center_logits = emo_outputs[center], target_logits = emo_outputs[pt[-1]]))
            else:
                target_.append('pad')
                target_label.append('pad')
                target_dist.append('None')
                self_emo_shift.append(0)
                cosine_similarity.append('None')
                kl_divergence.append('None')
                earth_mover_dist.append('None')

            if len(pp) != 0:
                opposite_.append(pp[-1])
                opposite_label.append(emo_dict[pp[-1]])
                opposite_dist.append(index - index_words.index(pp[-1]))
            else:
                opposite_.append('pad')
                opposite_label.append('pad')
                opposite_dist.append('None')

    return center_, target_, opposite_, center_label, target_label, opposite_label, target_dist, opposite_dist, self_emo_shift, cosine_similarity, kl_divergence, earth_mover_dist

def generate_interaction_data(dialog_dict, seq_dict, emo_dict, val_set, mode='context'):
    """Generate training/testing data (emo_train.csv & emo_test.csv) under specific modes.
    
    Args:
        mode:
            if mode == context: proposed transactional contexts, referred to IAAN.
            if mode == random: randomly sampled contexts, referred to baseline randIAAN.
    """
    center_train, target_train, opposite_train, center_label_train, target_label_train, opposite_label_train, target_dist_train, opposite_dist_train, self_emo_shift_train = [], [], [], [], [], [], [], [], []
    center_val, target_val, opposite_val, center_label_val, target_label_val, opposite_label_val, target_dist_val, opposite_dist_val, self_emo_shift_val = [], [], [], [], [], [], [], [], []
    cosine_similarity_train, kl_divergence_train, earth_mover_dist_train = [], [], []
    cosine_similarity_val, kl_divergence_val, earth_mover_dist_val = [], [], []
    if mode=='context':
        generator = generate_interaction_sample

    for k in dialog_dict.keys():
        dialog_order = dialog_dict[k]
        # training set
        if val_set not in k:
            c, t, o, cl, tl, ol, td, od, ses, cs, kld, emd = generator(dialog_order, seq_dict, emo_dict)
            center_train += c
            target_train += t
            opposite_train += o
            center_label_train += cl
            target_label_train += tl
            opposite_label_train += ol
            target_dist_train += td
            opposite_dist_train += od
            self_emo_shift_train += ses
            cosine_similarity_train += cs
            kl_divergence_train += kld
            earth_mover_dist_train += emd
        # validation set
        else:
            c, t, o, cl, tl, ol, td, od, ses, cs, kld, emd = generator(dialog_order, seq_dict, emo_dict, val=True)
            center_val += c
            target_val += t
            opposite_val += o
            center_label_val += cl
            target_label_val += tl
            opposite_label_val += ol
            target_dist_val += td
            opposite_dist_val += od
            self_emo_shift_val += ses
            cosine_similarity_val += cs
            kl_divergence_val += kld
            earth_mover_dist_val += emd

    # save dialog pairs to train.csv and test.csv
    train_filename= './data/emo_train.csv'
    val_filename= './data/emo_test.csv'
    column_order = ['center', 'target', 'opposite', 'center_label', 'target_label', 'opposite_label', 'target_dist', 'opposite_dist', 'self_emo_shift', 'cosine_similarity', 'kl_divergence', 'earth_mover_dist']
    # train
    d = {'center': center_train, 'target': target_train, 'opposite': opposite_train, 'center_label': center_label_train, 
         'target_label': target_label_train, 'opposite_label': opposite_label_train, 'target_dist': target_dist_train, 'opposite_dist': opposite_dist_train, 'self_emo_shift': self_emo_shift_train, 'cosine_similarity': cosine_similarity_train, 'kl_divergence': kl_divergence_train, 'earth_mover_dist': earth_mover_dist_train}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(train_filename, sep=',', index = False)
    # validation
    d = {'center': center_val, 'target': target_val, 'opposite': opposite_val, 'center_label': center_label_val, 
         'target_label': target_label_val, 'opposite_label': opposite_label_val, 'target_dist': target_dist_val, 'opposite_dist': opposite_dist_val, 'self_emo_shift': self_emo_shift_val, 'cosine_similarity': cosine_similarity_val, 'kl_divergence': kl_divergence_val, 'earth_mover_dist': earth_mover_dist_val}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(val_filename, sep=',', index = False)

def gen_train_test_pair(data_frame, X, Y, test_utt_name=None):
    for index, row in data_frame.iterrows():
        X.append([])
        center_utt_name = row[0]
        target_utt_name = row[1]
        oppo_utt_name = row[2]
        
        center_utt_feat = feat_pooled[center_utt_name]
        target_utt_feat = feat_pooled[target_utt_name]
        oppo_utt_feat = feat_pooled[oppo_utt_name]
        
        #target_utt_emo = emo_num_dict[row[4]]
        #oppo_utt_emo = emo_num_dict[row[5]]
        self_emo_shift = row[-1]
        
        #X[-1].append(np.concatenate((center_utt_feat.flatten(), target_utt_feat.flatten(), oppo_utt_feat.flatten())))
        if target_utt_name != 'pad':
            cos_sim = cal_cosine_similarity(center_logits=emo_outputs[center_utt_name], target_logits=emo_outputs[target_utt_name])
            kl_div = cal_kl_divergence(center_logits=emo_outputs[center_utt_name], target_logits=emo_outputs[target_utt_name])
            earth_mover_dist = cal_earth_mover_dist(center_logits=emo_outputs[center_utt_name], target_logits=emo_outputs[target_utt_name])
        else:
            cos_sim = 1
            kl_div = 0
            earth_mover_dist = 0
        #X[-1].append(np.concatenate((np.array([cos_sim, kl_div, earth_mover_dist]), center_utt_feat.flatten(), target_utt_feat.flatten(), oppo_utt_feat.flatten())))
        X[-1].append([cos_sim])
        if center_utt_name in four_type_utt_list:
            Y.append(self_emo_shift)

        if test_utt_name != None:
            test_utt_name.append(center_utt_name)
        
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

def my_custom_score(y_true, y_pred):
    UAR = recall_score(y_true, y_pred, average='macro')
    return UAR

if __name__ == "__main__":
    # dimension of each utterance: (n, 45)
    # n:number of time frames in the utterance
    emo_num_dict = {'ang': 0, 'hap': 1, 'neu':2, 'sad': 3, 'sur': 4, 'fru': 5, 'xxx': 6, 'oth': 7, 'fea': 8, 'dis': 9, 'pad': 10}
    feat_pooled = joblib.load('./data/feat_preprocessing.pkl')
    
    # label
    emo_all_dict = joblib.load('./data/emo_all.pkl')
    emo_shift_all_dict = joblib.load('./data/4emo_shift_all_rearrange.pkl')
    
    utt_logits_outputs_fold1 = joblib.load('./data/original_iaan/utt_logits_outputs_fold1.pkl')
    utt_logits_outputs_fold2 = joblib.load('./data/original_iaan/utt_logits_outputs_fold2.pkl')
    utt_logits_outputs_fold3 = joblib.load('./data/original_iaan/utt_logits_outputs_fold3.pkl')
    utt_logits_outputs_fold4 = joblib.load('./data/original_iaan/utt_logits_outputs_fold4.pkl')
    utt_logits_outputs_fold5 = joblib.load('./data/original_iaan/utt_logits_outputs_fold5.pkl')
    #emo_outputs = joblib.load('./data/rearrange_single_outputs_iaan.pkl')
    # check pretrained model performance
    preds, gts = [], []
    for utt in emo_all_dict:
        if emo_all_dict[utt] in ['ang', 'hap', 'neu', 'sad']:
            gts.append(emo_num_dict[emo_all_dict[utt]])
            if utt[4] == '1':
                preds.append(utt_logits_outputs_fold1[utt].argmax())
            elif utt[4] == '2':
                preds.append(utt_logits_outputs_fold2[utt].argmax())
            elif utt[4] == '3':
                preds.append(utt_logits_outputs_fold3[utt].argmax())
            elif utt[4] == '4':
                preds.append(utt_logits_outputs_fold4[utt].argmax())
            else:
                preds.append(utt_logits_outputs_fold5[utt].argmax())
    print('pretrained model UAR =', round(100*recall_score(gts, preds, average='macro'), 2), '%')
    print('pretrained model ACC =', round(100*accuracy_score(gts, preds), 2), '%')

    # dialog order
    #dialog_dict = joblib.load('./data/dialog_rearrange.pkl')
    dialog_dict = joblib.load('./data/dialog_rearrange_4emo_iemocap.pkl')
    
    val = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
    pred = []
    gt = []
    pred_prob_dict_fold1, pred_prob_dict_fold2, pred_prob_dict_fold3, pred_prob_dict_fold4, pred_prob_dict_fold5 = {}, {}, {}, {}, {}
    five_fold_best_threshold = []
    for val_ in val:
        if val_ == 'Ses01':
            emo_outputs = utt_logits_outputs_fold1
        elif val_ == 'Ses02':
            emo_outputs = utt_logits_outputs_fold2
        elif val_ == 'Ses03':
            emo_outputs = utt_logits_outputs_fold3
        elif val_ == 'Ses04':
            emo_outputs = utt_logits_outputs_fold4
        else:
            emo_outputs = utt_logits_outputs_fold5
        four_type_utt_list = [] # len:5531
        print("################{}################".format(val_))

        # generate training data/val data
        generate_interaction_data(dialog_dict, feat_pooled, emo_all_dict, val_set=val_)
        emo_train = pd.read_csv('./data/emo_train.csv')
        emo_test = pd.read_csv('./data/emo_test.csv')
        threshold_performance_from_train_dict = {}
        # training 
        #for t in np.arange(-1.000, 1.001, 0.001):
        for t in np.arange(0.000, 1.001, 0.001):
            print('########## t =', t, '##########')
            emo_shift_preds = []
            emo_shift_gts = []
            for i in range(len(emo_train)):
                if emo_train['cosine_similarity'][i] != 'None':
                    if float(emo_train['cosine_similarity'][i]) >= t:
                        emo_shift_preds.append(0.0)
                    else:
                        emo_shift_preds.append(1.0)
                else:
                    emo_shift_preds.append(0.0)
                emo_shift_gts.append(emo_shift_all_dict[emo_train['center'][i]])
            UAR = recall_score(emo_shift_gts, emo_shift_preds, average='macro')
            precision = precision_score(emo_shift_gts, emo_shift_preds)
            print('UAR:', round(UAR*100, 2), '%')
            print('precision (predcit label 1):', round(precision*100, 2), '%')
            UAR_weight = 0.0
            threshold_performance_from_train_dict[t] = UAR*UAR_weight + precision*(1 - UAR_weight)

        best_threshold_from_train = -2
        best_performance_from_train = -1
        for k in threshold_performance_from_train_dict:
            if threshold_performance_from_train_dict[k] > best_performance_from_train:
                best_performance_from_train = threshold_performance_from_train_dict[k]
                best_threshold_from_train = k
        print('The best threshold from train set:', best_threshold_from_train)
        five_fold_best_threshold.append(best_threshold_from_train)

        # testing
        for i in range(len(emo_test)):
            if emo_test['cosine_similarity'][i] != 'None':
                if float(emo_test['cosine_similarity'][i]) >= best_threshold_from_train:
                    pred.append(0.0)
                    pred_prob_for_dict = 0.0
                else:
                    pred.append(1.0)
                    pred_prob_for_dict = 1.0
            else:
                pred.append(0.0)
                pred_prob_for_dict = 0.0
            if emo_test['center'][i][4] == '1':
                pred_prob_dict_fold1[emo_test['center'][i]] = pred_prob_for_dict
            elif emo_test['center'][i][4] == '2':
                pred_prob_dict_fold2[emo_test['center'][i]] = pred_prob_for_dict
            elif emo_test['center'][i][4] == '3':
                pred_prob_dict_fold3[emo_test['center'][i]] = pred_prob_for_dict
            elif emo_test['center'][i][4] == '4':
                pred_prob_dict_fold4[emo_test['center'][i]] = pred_prob_for_dict
            else:
                pred_prob_dict_fold5[emo_test['center'][i]] = pred_prob_for_dict
            gt.append(emo_shift_all_dict[emo_test['center'][i]])
        # training outputs from the best threshold
        for i in range(len(emo_train)):
            if emo_train['cosine_similarity'][i] != 'None':
                if float(emo_train['cosine_similarity'][i]) >= best_threshold_from_train:
                    pred_prob_for_dict = 0.0
                else:
                    pred_prob_for_dict = 1.0
            else:
                pred_prob_for_dict = 0.0
            if val_ == 'Ses01':
                pred_prob_dict_fold1[emo_train['center'][i]] = pred_prob_for_dict
            elif val_ == 'Ses02':
                pred_prob_dict_fold2[emo_train['center'][i]] = pred_prob_for_dict
            elif val_ == 'Ses03':
                pred_prob_dict_fold3[emo_train['center'][i]] = pred_prob_for_dict
            elif val_ == 'Ses04':
                pred_prob_dict_fold4[emo_train['center'][i]] = pred_prob_for_dict
            elif val_ == 'Ses05':
                pred_prob_dict_fold5[emo_train['center'][i]] = pred_prob_for_dict
    print('## MODEL PERFORMANCE ##')
    print('UAR:', round(recall_score(gt, pred, average='macro')*100, 2), '%')
    print('UAR 2 type:', recall_score(gt, pred, average=None))
    print('precision 2 type:', precision_score(gt, pred, average=None))
    print(confusion_matrix(gt, pred))

    joblib.dump(pred_prob_dict_fold1, './output/DTC_fold1.pkl')
    joblib.dump(pred_prob_dict_fold2, './output/DTC_fold2.pkl')
    joblib.dump(pred_prob_dict_fold3, './output/DTC_fold3.pkl')
    joblib.dump(pred_prob_dict_fold4, './output/DTC_fold4.pkl')
    joblib.dump(pred_prob_dict_fold5, './output/DTC_fold5.pkl')
    
    print('5 fold the best threshold:', five_fold_best_threshold)
    