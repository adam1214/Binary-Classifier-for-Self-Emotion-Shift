import joblib
import numpy as np

# 把(n, 45)壓成兩個(1, 45), 一個取平均, 一個取std
feat_pooled = joblib.load('./feat_pooled.pkl')
feat_preprocessing = {}
for k in feat_pooled:
    ori_feat = feat_pooled[k]
    #feat_preprocessing[k] = np.mean(ori_feat, axis=0)
    feat_preprocessing[k] = np.vstack((np.mean(ori_feat, axis=0), np.std(ori_feat, axis=0)))
    if np.isnan(feat_preprocessing[k]).any() == True:
        print(k)
        #print(feat_pooled[k].shape)
        feat_preprocessing[k] = np.nan_to_num(feat_preprocessing[k])
joblib.dump(feat_preprocessing, 'feat_preprocessing.pkl')
    