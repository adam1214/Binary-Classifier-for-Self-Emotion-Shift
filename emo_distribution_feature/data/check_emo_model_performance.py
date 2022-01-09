import joblib
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
emo_all = joblib.load('./emo_all.pkl')
emo_output = joblib.load('./dag_outputs_4_all_fold_single_rearrange.pkl')
#emo_output = joblib.load('./rearrange_single_outputs_iaan.pkl')
emo_num_dict = {'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3}

preds, gts = [], []
for k in emo_output:
    if emo_all[k] in ['ang', 'hap', 'neu', 'sad']:
        preds.append(emo_output[k].argmax())
        gts.append(emo_num_dict[emo_all[k]])

print('UAR =', round(recall_score(gts, preds, average='macro')*100, 2), '%')
print('ACC =', round(accuracy_score(gts, preds)*100, 2), '%')