import joblib
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import pandas as pd
import pdb
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek, SMOTEENN
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')  # "error", "ignore", "always", "default", "module" or "once"

class trainData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index].long()
        
    def __len__ (self):
        return len(self.X_data)
    
class valData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index].long()
        
    def __len__ (self):
        return len(self.X_data)
    
class testData(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index].long()
        
    def __len__ (self):
        return len(self.X_data)

class binaryClassification(nn.Module):
    def __init__(self):
        super(binaryClassification, self).__init__()
        # Number of input features is 270.
        self.layer_1 = nn.Linear(271, 64) 
        self.layer_2 = nn.Linear(64, 32)
        self.layer_3 = nn.Linear(32, 16)
        self.layer_out = nn.Linear(16, 1) 
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.3)
        self.batchnorm1 = nn.BatchNorm1d(64)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.batchnorm3 = nn.BatchNorm1d(16)
        
    def forward(self, inputs):
        x = self.relu(self.layer_1(inputs))
        x = self.batchnorm1(x)
        x = self.relu(self.layer_2(x))
        x = self.batchnorm2(x)
        x = self.relu(self.layer_3(x))
        x = self.batchnorm3(x)
        x = self.dropout(x)
        x = self.layer_out(x)
        
        return x
    
def binary_uar(y_pred, y_test):
    y_pred_tag = torch.round(torch.sigmoid(y_pred))
    uar = round(recall_score(y_test.cpu(), y_pred_tag.long().cpu(), average='macro')*100, 2)

    return uar

def generate_interaction_sample(index_words, seq_dict, emo_dict):
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
        if emo_dict[center] in emo:
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

def generate_interaction_data(dialog_dict, seq_dict, emo_dict, test_set, val_set, mode='context'):
    """Generate training/testing data (emo_train.csv & emo_test.csv) under specific modes.
    
    Args:
        mode:
            if mode == context: proposed transactional contexts, referred to IAAN.
            if mode == random: randomly sampled contexts, referred to baseline randIAAN.
    """
    center_train, target_train, opposite_train, center_label_train, target_label_train, opposite_label_train, target_dist_train, opposite_dist_train, self_emo_shift_train = [], [], [], [], [], [], [], [], []
    center_val, target_val, opposite_val, center_label_val, target_label_val, opposite_label_val, target_dist_val, opposite_dist_val, self_emo_shift_val = [], [], [], [], [], [], [], [], []
    center_test, target_test, opposite_test, center_label_test, target_label_test, opposite_label_test, target_dist_test, opposite_dist_test, self_emo_shift_test = [], [], [], [], [], [], [], [], []
    if mode=='context':
        generator = generate_interaction_sample

    for k in dialog_dict.keys():
        dialog_order = dialog_dict[k]
        # training set
        if test_set not in k and val_set not in k:
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
        # val set
        elif val_set in k:
            c, t, o, cl, tl, ol, td, od, ses = generator(dialog_order, seq_dict, emo_dict)
            center_val += c
            target_val += t
            opposite_val += o
            center_label_val += cl
            target_label_val += tl
            opposite_label_val += ol
            target_dist_val += td
            opposite_dist_val += od
            self_emo_shift_val += ses
        # test set
        else:
            c, t, o, cl, tl, ol, td, od, ses = generator(dialog_order, seq_dict, emo_dict)
            center_test += c
            target_test += t
            opposite_test += o
            center_label_test += cl
            target_label_test += tl
            opposite_label_test += ol
            target_dist_test += td
            opposite_dist_test += od
            self_emo_shift_test += ses

    # save dialog pairs to train.csv and test.csv
    train_filename = './data/emo_train.csv'
    val_filename = './data/emo_val.csv'
    test_filename = './data/emo_test.csv'
    column_order = ['center', 'target', 'opposite', 'center_label', 'target_label', 'opposite_label', 'target_dist', 'opposite_dist', 'self_emo_shift']
    # train
    d = {'center': center_train, 'target': target_train, 'opposite': opposite_train, 'center_label': center_label_train, 
         'target_label': target_label_train, 'opposite_label': opposite_label_train, 'target_dist': target_dist_train, 'opposite_dist': opposite_dist_train, 'self_emo_shift': self_emo_shift_train}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(train_filename, sep=',', index = False)
    # val
    d = {'center': center_val, 'target': target_val, 'opposite': opposite_val, 'center_label': center_label_val, 
         'target_label': target_label_val, 'opposite_label': opposite_label_val, 'target_dist': target_dist_val, 'opposite_dist': opposite_dist_val, 'self_emo_shift': self_emo_shift_val}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(val_filename, sep=',', index = False)
    # test
    d = {'center': center_test, 'target': target_test, 'opposite': opposite_test, 'center_label': center_label_test, 
         'target_label': target_label_test, 'opposite_label': opposite_label_test, 'target_dist': target_dist_test, 'opposite_dist': opposite_dist_test, 'self_emo_shift': self_emo_shift_test}
    df = pd.DataFrame(data=d)
    df[column_order].to_csv(test_filename, sep=',', index = False)

def gen_train_val_test(data_frame, X, Y):
    for index, row in data_frame.iterrows():
        X.append([])
        center_utt_name = row[0]
        target_utt_name = row[1]
        oppo_utt_name = row[2]
        
        center_utt_feat = feat_pooled[center_utt_name]
        target_utt_feat = feat_pooled[target_utt_name]
        oppo_utt_feat = feat_pooled[oppo_utt_name]
        if target_utt_name == 'pad':
            center_target_duration = 0.
        else:
            center_target_duration = utt_time_dict[center_utt_name]['start_sec'] - utt_time_dict[target_utt_name]['start_sec']
        
        center_utt_feat = feat_pooled[center_utt_name]
        target_utt_feat = feat_pooled[target_utt_name]
        oppo_utt_feat = feat_pooled[oppo_utt_name]
        
        #target_utt_emo = emo_num_dict[row[4]]
        #oppo_utt_emo = emo_num_dict[row[5]]
        self_emo_shift = row[-1]
        
        X[-1].append(np.concatenate((center_utt_feat.flatten(), target_utt_feat.flatten(), oppo_utt_feat.flatten(), np.array(center_target_duration).flatten())))
        Y.append(self_emo_shift)

def model_pred_and_gt(y_pred_list, y_gt_list, loader, model):
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            y_pred = model(X_batch)
            y_pred = torch.sigmoid(y_pred)
            y_pred_tag = torch.round(y_pred).long()
            try:
                y_pred_list += y_pred_tag.cpu().squeeze().tolist()
                y_gt_list += y_batch.tolist()
            except:
                y_pred_list.append(y_pred_tag.cpu().squeeze().tolist())
                y_gt_list.append(y_batch.tolist())

if __name__ == "__main__":
    BATCH_SIZE = 32
    LEARNING_RATE = 0.00005
    WEIGHT_DECAY = 0.01
    EPOCH = 100
    
    utt_time_dict = joblib.load('./data/utt_time_dict.pkl')
    # dimension of each utterance: (n, 45)
    # n:number of time frames in the utterance
    torch.manual_seed(100)
    
    emo_num_dict = {'ang': 0, 'hap': 1, 'neu':2, 'sad': 3, 'sur': 4, 'fru': 5, 'xxx': 6, 'oth': 7, 'fea': 8, 'dis': 9, 'pad': 10}
    feat_pooled = joblib.load('./data/feat_preprocessing.pkl')
    
    # label
    emo_all_dict = joblib.load('./data/emo_all.pkl')
    
    # dialog order
    dialog_dict = joblib.load('./data/dialog.pkl')
    
    test = ['Ses01', 'Ses02', 'Ses03', 'Ses04', 'Ses05']
    val = ['Ses02', 'Ses03', 'Ses01', 'Ses05', 'Ses04']
    pred = []
    gt = []
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    
    for test_, val_ in zip(test, val):
        print("################{}################".format(test_))
        
        model = binaryClassification()
        model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
        
        
        # generate training data/val data/ test data
        generate_interaction_data(dialog_dict, feat_pooled, emo_all_dict, test_set=test_, val_set=val_)
        emo_train = pd.read_csv('./data/emo_train.csv')
        emo_val = pd.read_csv('./data/emo_val.csv')
        emo_test = pd.read_csv('./data/emo_test.csv')
        
        train_X, train_Y, val_X, val_Y, test_X, test_Y = [], [], [], [], [], []
        
        gen_train_val_test(emo_train, train_X, train_Y)
        train_X = np.array(train_X)
        train_X = train_X.squeeze(1)
        train_data = trainData(torch.FloatTensor(train_X), torch.FloatTensor(train_Y))
        train_loader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
        
        gen_train_val_test(emo_val, val_X, val_Y)
        val_X = np.array(val_X)
        val_X = val_X.squeeze(1)
        val_data = valData(torch.FloatTensor(val_X), torch.FloatTensor(val_Y))
        val_loader = DataLoader(dataset=val_data, batch_size=BATCH_SIZE)
        
        gen_train_val_test(emo_test, test_X, test_Y)
        test_X = np.array(test_X)
        test_X = test_X.squeeze(1)
        test_data = testData(torch.FloatTensor(test_X), torch.FloatTensor(test_Y))
        test_loader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE)
        
        counter = Counter(train_Y)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(counter[0]/counter[1]).to(device))
        
        # training
        val_uar_list = []
        max_val_uar = 0
        best_epoch = 0
        for e in range(1, EPOCH+1, 1):
            model.train()
            epoch_loss = 0
            epoch_uar = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                
                y_pred = model(X_batch)
                uar = binary_uar(y_pred, y_batch.unsqueeze(1))
                loss = criterion(y_pred, y_batch.unsqueeze(1).float())
                
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                epoch_uar += uar.item()
                
            # validation
            y_pred_list_validation = []
            y_gt_list_validation = []
            model.eval()
            model_pred_and_gt(y_pred_list_validation, y_gt_list_validation, val_loader, model)
            val_uar = recall_score(y_gt_list_validation, y_pred_list_validation, average='macro')*100
            val_uar_list.append(val_uar)
            if val_uar > max_val_uar:
                max_val_uar = val_uar
                best_epoch = e
                checkpoint = {'epoch': e, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'loss': loss}
                torch.save(checkpoint, './model/mlp_pytorch_best_model.pth')
            print(f'Epoch {e+0:03}: | Loss: {epoch_loss/len(train_loader):.5f} | train_uar: {epoch_uar/len(train_loader):.3f} | val_uar: {val_uar:.2f}')
            print(confusion_matrix(y_gt_list_validation, y_pred_list_validation))
        print('The best epoch:', best_epoch)

        # testing
        y_pred_list_testing = []
        model = binaryClassification()
        model.to(device)
        checkpoint = torch.load('./model/mlp_pytorch_best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        model_pred_and_gt(pred, gt, test_loader, model)
        
    print('UAR:', round(recall_score(gt, pred, average='macro')*100, 2), '%')
    #print('ACC:', round(accuracy_score(gt, pred)*100, 2), '%')
    print('precision (predcit label 1):', round(precision_score(gt, pred)*100, 2), '%')
    print(confusion_matrix(gt, pred))