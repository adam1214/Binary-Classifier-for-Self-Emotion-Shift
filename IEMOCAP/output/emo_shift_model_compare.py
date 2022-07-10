import joblib
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import scipy.stats
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
import pdb

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
            '''
            if emo_dict[center] in emo:
                four_type_utt_list.append(center)
            '''
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
                cosine_similarity.append(1)
                kl_divergence.append(0)
                earth_mover_dist.append(0)

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
    #center_val, target_val, opposite_val, center_label_val, target_label_val, opposite_label_val, target_dist_val, opposite_dist_val, self_emo_shift_val = [], [], [], [], [], [], [], [], []
    cosine_similarity_train, kl_divergence_train, earth_mover_dist_train = [], [], []
    #cosine_similarity_val, kl_divergence_val, earth_mover_dist_val = [], [], []
    if mode=='context':
        generator = generate_interaction_sample

    for k in dialog_dict.keys():
        dialog_order = dialog_dict[k]
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

    # save dialog pairs to train.csv and test.csv
    train_filename= './emo_all.csv'
    column_order = ['center', 'target', 'opposite', 'center_label', 'target_label', 'opposite_label', 'target_dist', 'opposite_dist', 'self_emo_shift', 'cosine_similarity', 'kl_divergence', 'earth_mover_dist']
    # train
    d = {'center': center_train, 'target': target_train, 'opposite': opposite_train, 'center_label': center_label_train, 
         'target_label': target_label_train, 'opposite_label': opposite_label_train, 'target_dist': target_dist_train, 'opposite_dist': opposite_dist_train, 'self_emo_shift': self_emo_shift_train, 'cosine_similarity': cosine_similarity_train, 'kl_divergence': kl_divergence_train, 'earth_mover_dist': earth_mover_dist_train}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(train_filename, sep=',', index = False)

if __name__ == '__main__':
    utt_logits_outputs_fold1 = joblib.load('../data/original_iaan/utt_logits_outputs_fold1.pkl')
    utt_logits_outputs_fold2 = joblib.load('../data/original_iaan/utt_logits_outputs_fold2.pkl')
    utt_logits_outputs_fold3 = joblib.load('../data/original_iaan/utt_logits_outputs_fold3.pkl')
    utt_logits_outputs_fold4 = joblib.load('../data/original_iaan/utt_logits_outputs_fold4.pkl')
    utt_logits_outputs_fold5 = joblib.load('../data/original_iaan/utt_logits_outputs_fold5.pkl')
    
    emo_all = joblib.load('../data/emo_all.pkl')
    
    emo_shift_all = joblib.load('../data/4emo_shift_all_rearrange.pkl')
    iaan_emo_shift = joblib.load('./demo_model/iaan_emo_shift_output_precision1.pkl')
    svm_emo_shift = joblib.load('./demo_model/SVM_emo_shift_output.pkl')
    rf_emo_shift = joblib.load('./demo_model/RandomForest_emo_shift_output.pkl')
    LogisticRegression_emo_shift = joblib.load('./demo_model/LogisticRegression_emo_shift_output.pkl')
    dnn_emo_prob_fold1 = joblib.load('./demo_model/MLPPytorch_emo_shift_output_fold1.pkl')
    dnn_emo_prob_fold2 = joblib.load('./demo_model/MLPPytorch_emo_shift_output_fold2.pkl')
    dnn_emo_prob_fold3 = joblib.load('./demo_model/MLPPytorch_emo_shift_output_fold3.pkl')
    dnn_emo_prob_fold4 = joblib.load('./demo_model/MLPPytorch_emo_shift_output_fold4.pkl')
    dnn_emo_prob_fold5 = joblib.load('./demo_model/MLPPytorch_emo_shift_output_fold5.pkl')
    dnn_emo_shift = {}
    for utt in dnn_emo_prob_fold1:
        if utt[4] == '1':
            dnn_emo_shift[utt] = dnn_emo_prob_fold1[utt]
        elif utt[4] == '2':
            dnn_emo_shift[utt] = dnn_emo_prob_fold2[utt]
        elif utt[4] == '3':
            dnn_emo_shift[utt] = dnn_emo_prob_fold3[utt]
        elif utt[4] == '4':
            dnn_emo_shift[utt] = dnn_emo_prob_fold4[utt]
        elif utt[4] == '5':
            dnn_emo_shift[utt] = dnn_emo_prob_fold5[utt]
    
    feat_pooled = joblib.load('../data/feat_preprocessing.pkl')
    dialog_dict = joblib.load('../data/dialog_rearrange_4emo_iemocap.pkl')
    emo_num_map = {'ang':0, 'hap':1, 'neu':2, 'sad':3}
    
    emo_outputs = {}
    for utt in emo_all:
        if utt[4] == '1':
            emo_outputs[utt] = utt_logits_outputs_fold1[utt]
        elif utt[4] == '2':
            emo_outputs[utt] = utt_logits_outputs_fold2[utt]
        elif utt[4] == '3':
            emo_outputs[utt] = utt_logits_outputs_fold3[utt]
        elif utt[4] == '4':
            emo_outputs[utt] = utt_logits_outputs_fold4[utt]
        elif utt[4] == '5':
            emo_outputs[utt] = utt_logits_outputs_fold5[utt]
    
    preds, labels = [], []
    for utt in emo_all:
        if emo_all[utt] in ['ang', 'hap', 'neu', 'sad']:
            labels.append(emo_num_map[emo_all[utt]])
            preds.append(emo_outputs[utt].argmax())
        
    
    print('##### CHECK IAAN PERFORMANCE')
    print('UAR:', round(recall_score(labels, preds, average='macro')*100, 2))
    print('ACC:', round(accuracy_score(labels, preds)*100, 2))
    
    generate_interaction_data(dialog_dict, feat_pooled, emo_all, val_set=None)
    emo_all_pd = pd.read_csv('./emo_all.csv')
    
    name_list = ['IAAN_EMO_SHIFT', 'SVM', 'Random Forest', 'Logistic Regression', 'DNN']
    model_prob_list = [iaan_emo_shift, svm_emo_shift, rf_emo_shift, LogisticRegression_emo_shift, dnn_emo_shift]
    model_performance_list = []
    for model_prob in model_prob_list:
        p, g = [], []
        for utt in model_prob:
            if model_prob[utt] > 0.5:
                p.append(1)
            else:
                p.append(0)
            g.append(int(emo_shift_all[utt]))
        print('## EMO_SHIFT MODEL PERFORMANCE ##')
        print(len(p), len(g))
        '''
        print('UAR:', round(recall_score(g, p, average='macro')*100, 2), '%')
        print('UAR 2 type:', recall_score(g, p, average=None))
        print('precision 2 type:', precision_score(g, p, average=None))
        print(confusion_matrix(g, p))
        '''
        result_string = 'Recall_0:' + str(round(recall_score(g, p, average=None)[0]*100, 2)) + '%\n' + 'Recall_1:' + str(round(recall_score(g, p, average=None)[1]*100, 2)) + '%\n' + 'Precision_0:' + str(round(precision_score(g, p, average=None)[0]*100, 2)) + '%\n' + 'Precision_1:' + str(round(precision_score(g, p, average=None)[1]*100, 2)) + '%'
        model_performance_list.append(result_string)
    for name, model_prob, model_performance in zip(name_list, model_prob_list, model_performance_list):
        plt.figure()
        cos_sim_list_no_shift = []
        cos_sim_list_shift = []
        cos_sim_list_no_shift_gt = []
        cos_sim_list_shift_gt = []
        for i, row in emo_all_pd.iterrows():
            #print(row['center'], row['cosine_similarity'])
            if model_prob[row['center']] > 0.5:
                cos_sim_list_shift.append(row['cosine_similarity'])
            else:
                cos_sim_list_no_shift.append(row['cosine_similarity'])
                
            if emo_shift_all[row['center']] > 0.5:
                cos_sim_list_shift_gt.append(row['cosine_similarity'])
            else:
                cos_sim_list_no_shift_gt.append(row['cosine_similarity'])
        sns.histplot(np.array(cos_sim_list_shift), bins=np.arange(-1, 1.1, 0.1), color='red', kde=True, stat='probability', fill=False, element="bars")
        sns.histplot(np.array(cos_sim_list_no_shift), bins=np.arange(-1, 1.1, 0.1), color='blue', kde=True, stat='probability', fill=False, element="bars")
        sns.histplot(np.array(cos_sim_list_shift_gt), bins=np.arange(-1, 1.1, 0.1), color='gold', kde=True, stat='probability', fill=False, element="bars")
        sns.histplot(np.array(cos_sim_list_no_shift_gt), bins=np.arange(-1, 1.1, 0.1), color='green', kde=True, stat='probability', fill=False, element="bars")
        plt.legend(["Model predict emo. shift", "Model predict emo. no shift", "Label of emo. shift", "Label of emo. no shift"], loc='best')
        plt.title(name)
        plt.xlabel('cosine_similarity')
        plt.text(x=-1, y=0.2 , s=model_performance)
        plt.savefig('./demo_model/' + name + ".png")
    