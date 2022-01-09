import joblib
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
emo_shift_all = joblib.load('./emo_shift_all_rearrange.pkl')
emo_all = joblib.load('./emo_all.pkl')
iaan_emo_shift_output = joblib.load('./iaan_emo_shift_output_rearrange.pkl')

preds, gts = [], []
for k in iaan_emo_shift_output:
    if emo_all[k] in ['ang', 'hap', 'neu', 'sad']:
        if iaan_emo_shift_output[k] > 0.5:
            preds.append(1)
        else:
            preds.append(0)
        gts.append(int(emo_shift_all[k]))

print('UAR =', round(recall_score(gts, preds, average='macro')*100, 2), '%')
print('precision =', round(precision_score(gts, preds)*100, 2), '%')