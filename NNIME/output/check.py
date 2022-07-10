import joblib

output1 = joblib.load('./DecisionTreeClassifier_emo_shift_output_0_0.pkl')
output2 = joblib.load('./DecisionTreeClassifier_emo_shift_output_0_4.pkl')

for k in output1:
    if output1[k] != output2[k]:
        print(123456)