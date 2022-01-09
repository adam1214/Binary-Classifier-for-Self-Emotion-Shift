import joblib
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import pandas as pd
import pdb
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler 
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTETomek, SMOTEENN

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
        
        X[-1].append(np.concatenate((center_utt_feat.flatten(), target_utt_feat.flatten(), oppo_utt_feat.flatten())))
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

if __name__ == "__main__":
    # dimension of each utterance: (n, 45)
    # n:number of time frames in the utterance
    emo_num_dict = {'ang': 0, 'hap': 1, 'neu':2, 'sad': 3, 'sur': 4, 'fru': 5, 'xxx': 6, 'oth': 7, 'fea': 8, 'dis': 9, 'pad': 10}
    feat_pooled = joblib.load('./data/feat_preprocessing.pkl')
    
    # label
    emo_all_dict = joblib.load('./data/emo_all.pkl')
    
    # dialog order
    dialog_dict = joblib.load('./data/dialog.pkl')
    
    val = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
    pred = []
    gt = []
    pred_prob_dict = {}
    for val_ in val:
        four_type_utt_list = [] # len:5531
        print("################{}################".format(val_))
        
        train_X, train_Y, test_X, test_Y = [], [], [], []
        test_utt_name = []

        # generate training data/val data
        generate_interaction_data(dialog_dict, feat_pooled, emo_all_dict, val_set=val_)
        emo_train = pd.read_csv('./data/emo_train.csv')
        emo_test = pd.read_csv('./data/emo_test.csv')
        
        gen_train_test_pair(emo_train, train_X, train_Y)
        #X_upsample, Y_upsample = upsampling(train_X, train_Y)
        
        train_X = np.array(train_X)
        train_X = train_X.squeeze(1)
        
        counter = Counter(train_Y)
        clf = make_pipeline(LogisticRegression(random_state=100, max_iter=1000, multi_class = 'multinomial', n_jobs = -1, class_weight= {0: 1/counter[0], 1: 1/counter[1]}))
        #clf.fit(X_upsample, Y_upsample)
        clf.fit(train_X, train_Y)
        
        # testing
        gen_train_test_pair(emo_test, test_X, test_Y, test_utt_name)
        test_X = np.array(test_X)
        test_X = test_X.squeeze(1)
        #p = clf.predict(test_X)
        pred_prob_np = clf.predict_proba(test_X)
        p = []
        
        #pred += p.tolist()
        gt += test_Y
        for i, utt_name in enumerate(test_utt_name):
            pred_prob_dict[utt_name] = pred_prob_np[i][1]
            if utt_name in four_type_utt_list:
                if pred_prob_np[i][1] > 0.5:
                    p.append(1)
                else:
                    p.append(0)
        pred += p

    print('UAR:', round(recall_score(gt, pred, average='macro')*100, 2), '%')
    #print('ACC:', round(accuracy_score(gt, pred)*100, 2), '%')
    print('precision (predcit label 1):', round(precision_score(gt, pred)*100, 2), '%')
    print(confusion_matrix(gt, pred))

    joblib.dump(pred_prob_dict, './output/LogisticRegression_emo_shift_output.pkl')