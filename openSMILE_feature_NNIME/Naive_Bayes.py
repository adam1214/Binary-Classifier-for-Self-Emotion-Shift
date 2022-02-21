import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB, ComplementNB, CategoricalNB
import pandas as pd
import pdb
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler 
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek, SMOTEENN
import argparse
from argparse import RawTextHelpFormatter
from sklearn.metrics.pairwise import cosine_similarity
import scipy.stats

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
    train_filename= './data/emo_train.csv'
    val_filename= './data/emo_test.csv'
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

def gen_train_test_pair(data_frame, X, Y, utt_name=None):
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
        
        hc_hp_cos_sim = cal_cosine_similarity(center_logits=utt_hc[center_utt_name], target_logits=utt_hp[target_utt_name], dim=512)
        hc_hp_kl_div = cal_kl_divergence(center_logits=utt_hc[center_utt_name], target_logits=utt_hp[target_utt_name])
        hc_hp_earth_mover_dist = cal_earth_mover_dist(center_logits=utt_hc[center_utt_name], target_logits=utt_hp[target_utt_name])
        
        hc_hr_cos_sim = cal_cosine_similarity(center_logits=utt_hc[center_utt_name], target_logits=utt_hr[oppo_utt_name], dim=512)
        hc_hr_kl_div = cal_kl_divergence(center_logits=utt_hc[center_utt_name], target_logits=utt_hr[oppo_utt_name])
        hc_hr_earth_mover_dist = cal_earth_mover_dist(center_logits=utt_hc[center_utt_name], target_logits=utt_hr[oppo_utt_name])
        
        if target_utt_name != 'pad':
            Ec_Ep_cos_sim = cal_cosine_similarity(center_logits=emo_outputs[center_utt_name], target_logits=emo_outputs[target_utt_name], dim=4)
            Ec_Ep_kl_div = cal_kl_divergence(center_logits=emo_outputs[center_utt_name], target_logits=emo_outputs[target_utt_name])
            Ec_Ep_earth_mover_dist = cal_earth_mover_dist(center_logits=emo_outputs[center_utt_name], target_logits=emo_outputs[target_utt_name])
        else:
            Ec_Ep_cos_sim = 1
            Ec_Ep_kl_div = 0
            Ec_Ep_earth_mover_dist = 0
            
        if oppo_utt_name != 'pad':
            Ec_Er_cos_sim = cal_cosine_similarity(center_logits=emo_outputs[center_utt_name], target_logits=emo_outputs[oppo_utt_name], dim=4)
            Ec_Er_kl_div = cal_kl_divergence(center_logits=emo_outputs[center_utt_name], target_logits=emo_outputs[oppo_utt_name])
            Ec_Er_earth_mover_dist = cal_earth_mover_dist(center_logits=emo_outputs[center_utt_name], target_logits=emo_outputs[oppo_utt_name])
        else:
            Ec_Er_cos_sim = 1
            Ec_Er_kl_div = 0
            Ec_Er_earth_mover_dist = 0

        X[-1].append(np.concatenate([np.array([Ec_Ep_cos_sim])]))
        if center_utt_name in four_type_utt_list:
            Y.append(self_emo_shift)

        if utt_name != None:
            utt_name.append(center_utt_name)
        
def upsampling(X, Y):
    counter = Counter(Y)
    print(counter)
    
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
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('-r', "--random_num", type=int, help="select random number?", default=100)
    args = parser.parse_args()
    # dimension of each utterance: (n, 45)
    # n:number of time frames in the utterance
    emo_num_dict = {'ang': 0, 'hap': 1, 'neu':2, 'sad': 3, 'sur': 4, 'fru': 5, 'xxx': 6, 'oth': 7, 'fea': 8, 'dis': 9, 'pad': 10}
    feat_pooled = joblib.load('./data/feat_preprocessing.pkl')
    
    # label
    emo_all_dict = joblib.load('./data/emo_all.pkl')
    
    utt_logits_outputs_fold1 = joblib.load('./data/original_iaan/utt_logits_outputs_fold1.pkl')
    utt_logits_outputs_fold2 = joblib.load('./data/original_iaan/utt_logits_outputs_fold2.pkl')
    utt_logits_outputs_fold3 = joblib.load('./data/original_iaan/utt_logits_outputs_fold3.pkl')
    utt_logits_outputs_fold4 = joblib.load('./data/original_iaan/utt_logits_outputs_fold4.pkl')
    utt_logits_outputs_fold5 = joblib.load('./data/original_iaan/utt_logits_outputs_fold5.pkl')
    
    utt_hc_fold1 = joblib.load('./data/original_iaan/utt_hc_fold1.pkl')
    utt_hc_fold2 = joblib.load('./data/original_iaan/utt_hc_fold2.pkl')
    utt_hc_fold3 = joblib.load('./data/original_iaan/utt_hc_fold3.pkl')
    utt_hc_fold4 = joblib.load('./data/original_iaan/utt_hc_fold4.pkl')
    utt_hc_fold5 = joblib.load('./data/original_iaan/utt_hc_fold5.pkl')
    
    utt_hp_fold1 = joblib.load('./data/original_iaan/utt_hp_fold1.pkl')
    utt_hp_fold2 = joblib.load('./data/original_iaan/utt_hp_fold2.pkl')
    utt_hp_fold3 = joblib.load('./data/original_iaan/utt_hp_fold3.pkl')
    utt_hp_fold4 = joblib.load('./data/original_iaan/utt_hp_fold4.pkl')
    utt_hp_fold5 = joblib.load('./data/original_iaan/utt_hp_fold5.pkl')
    
    utt_hr_fold1 = joblib.load('./data/original_iaan/utt_hr_fold1.pkl')
    utt_hr_fold2 = joblib.load('./data/original_iaan/utt_hr_fold2.pkl')
    utt_hr_fold3 = joblib.load('./data/original_iaan/utt_hr_fold3.pkl')
    utt_hr_fold4 = joblib.load('./data/original_iaan/utt_hr_fold4.pkl')
    utt_hr_fold5 = joblib.load('./data/original_iaan/utt_hr_fold5.pkl')
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
    pred_prob_dict_fold1 = {}
    pred_prob_dict_fold2 = {}
    pred_prob_dict_fold3 = {}
    pred_prob_dict_fold4 = {}
    pred_prob_dict_fold5 = {}
    for val_ in val:
        if val_ == 'Ses01':
            emo_outputs = utt_logits_outputs_fold1
            utt_hc = utt_hc_fold1
            utt_hp = utt_hp_fold1
            utt_hr = utt_hr_fold1
        elif val_ == 'Ses02':
            emo_outputs = utt_logits_outputs_fold2
            utt_hc = utt_hc_fold2
            utt_hp = utt_hp_fold2
            utt_hr = utt_hr_fold2
        elif val_ == 'Ses03':
            emo_outputs = utt_logits_outputs_fold3
            utt_hc = utt_hc_fold3
            utt_hp = utt_hp_fold3
            utt_hr = utt_hr_fold3
        elif val_ == 'Ses04':
            emo_outputs = utt_logits_outputs_fold4
            utt_hc = utt_hc_fold4
            utt_hp = utt_hp_fold4
            utt_hr = utt_hr_fold4
        elif val_ == 'Ses05':
            emo_outputs = utt_logits_outputs_fold5
            utt_hc = utt_hc_fold5
            utt_hp = utt_hp_fold5
            utt_hr = utt_hr_fold5
        utt_hp['pad'] = np.zeros(512)
        utt_hr['pad'] = np.zeros(512)
        '''
        for utt in emo_outputs:
            if utt[4] != val_[-1] and emo_all_dict[utt] in ['ang', 'hap', 'neu', 'sad']: # change to onehot label encoding
                emo_outputs[utt] = np.array([0., 0., 0., 0.], np.float32)
                emo_outputs[utt][emo_num_dict[emo_all_dict[utt]]] = 1.
        '''
        
        four_type_utt_list = [] # len:5531
        four_type_utt_list = [] # len:5531
        print("################{}################".format(val_))
        
        train_X, train_Y, test_X, test_Y = [], [], [], []
        train_utt_name, test_utt_name = [], []

        # generate training data/val data
        generate_interaction_data(dialog_dict, feat_pooled, emo_all_dict, val_set=val_)
        emo_train = pd.read_csv('./data/emo_train.csv')
        emo_test = pd.read_csv('./data/emo_test.csv')
        
        gen_train_test_pair(emo_train, train_X, train_Y, train_utt_name)
        X_upsample, Y_upsample = upsampling(train_X, train_Y)
        
        train_X = np.array(train_X)
        train_X = train_X.squeeze(1)
        '''
        clf = SVC(random_state=123, probability=True)
        scorer = make_scorer(my_custom_score, greater_is_better=True)
        params_space = {
            'C': [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 
            'degree': [3, 4, 5],
            'gamma': ['scale', 'auto'],
            'coef0': [-3, -2.5, -2, -1.5, -1, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
            'shrinking': [True, False],
            'tol': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
            'decision_function_shape': ['ovo', 'ovr'],
            'break_ties': [True, False]
        }
        s_CV = RandomizedSearchCV(clf, params_space, cv=5, verbose=1, n_jobs=-1, n_iter=20, scoring=scorer, refit=True, random_state=123)
        s_CV.fit(X_upsample, Y_upsample)
        CV_result = s_CV.cv_results_
        best_clf = s_CV.best_estimator_
        '''
        clf = make_pipeline(BernoulliNB())
        clf.fit(X_upsample, Y_upsample)
        # testing
        gen_train_test_pair(emo_test, test_X, test_Y, test_utt_name)
        test_X = np.array(test_X)
        test_X = test_X.squeeze(1)
        #p = clf.predict(test_X)
        #pred_prob_np = best_clf.predict_proba(test_X)
        pred_prob_np = clf.predict_proba(test_X)
        p = []
        
        #pred += p.tolist()
        gt += test_Y
        for i, utt_name in enumerate(test_utt_name):
            if val_ == 'Ses01':
                pred_prob_dict_fold1[utt_name] = pred_prob_np[i][1]
            elif val_ == 'Ses02':
                pred_prob_dict_fold2[utt_name] = pred_prob_np[i][1]
            elif val_ == 'Ses03':
                pred_prob_dict_fold3[utt_name] = pred_prob_np[i][1]
            elif val_ == 'Ses04':
                pred_prob_dict_fold4[utt_name] = pred_prob_np[i][1]
            elif val_ == 'Ses05':
                pred_prob_dict_fold5[utt_name] = pred_prob_np[i][1]
                
            if utt_name in four_type_utt_list:
                if pred_prob_np[i][1] > 0.5:
                    p.append(1)
                else:
                    p.append(0)
        pred += p
        
        # training output
        pred_prob_np = clf.predict_proba(train_X)
        for i, utt_name in enumerate(train_utt_name):
           if val_ == 'Ses01':
               pred_prob_dict_fold1[utt_name] = pred_prob_np[i][1]
           elif val_ == 'Ses02':
               pred_prob_dict_fold2[utt_name] = pred_prob_np[i][1]
           elif val_ == 'Ses03':
               pred_prob_dict_fold3[utt_name] = pred_prob_np[i][1]
           elif val_ == 'Ses04':
               pred_prob_dict_fold4[utt_name] = pred_prob_np[i][1]
           elif val_ == 'Ses05':
               pred_prob_dict_fold5[utt_name] = pred_prob_np[i][1]

    print('## MODEL PERFORMANCE ##')
    print(len(gt), len(pred))
    print('UAR:', round(recall_score(gt, pred, average='macro')*100, 2), '%')
    print('UAR 2 type:', recall_score(gt, pred, average=None))
    print('precision 2 type:', precision_score(gt, pred, average=None))
    print(confusion_matrix(gt, pred))

    joblib.dump(pred_prob_dict_fold1, './output/NB_emo_shift_output_fold1.pkl')
    joblib.dump(pred_prob_dict_fold2, './output/NB_emo_shift_output_fold2.pkl')
    joblib.dump(pred_prob_dict_fold3, './output/NB_emo_shift_output_fold3.pkl')
    joblib.dump(pred_prob_dict_fold4, './output/NB_emo_shift_output_fold4.pkl')
    joblib.dump(pred_prob_dict_fold5, './output/NB_emo_shift_output_fold5.pkl')
    '''
    path = 'uar.txt'
    f = open(path, 'a')
    f.write(str(recall_score(gt, pred, average='macro')*100)+'\n')
    f.close()
    
    path = 'precision.txt'
    f = open(path, 'a')
    f.write(str(precision_score(gt, pred)*100)+'\n')
    f.close()
    '''