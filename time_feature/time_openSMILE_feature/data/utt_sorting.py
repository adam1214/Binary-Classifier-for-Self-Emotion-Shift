import joblib

class utt_info:
    def __init__(self, utt_name, start_sec):
        self.utt_name = utt_name
        self.start_sec = start_sec
    def __repr__(self):
        return repr((self.utt_name, self.start_sec))

f = open('AllTranscriptions.txt', 'r')
dialog_rearrange = {}
utts_list = []
dialog_name = ''

for line in f.readlines():
    line_list = line.split()
    try:
        if line_list[1][0] == '[' and line_list[1][-1] == ':' and 'X' not in line_list[0]:
            time_str = line_list[1]
            #print(time_str, line_list[0], float(time_str[1:9]), float(time_str[10:18]))
            utts_list.append(utt_info(line_list[0], float(time_str[1:9])))
    except:
        #print(line_list)
        if line[0] == 'S':
            utts_list = sorted(utts_list, key = lambda s: s.start_sec)
            for utt_item in utts_list:
                dialog_rearrange[dialog_name].append(utt_item.utt_name)
            
            dialog_name = line[0:-5]
            dialog_rearrange[dialog_name] = []
            utts_list.clear()
        continue

utts_list = sorted(utts_list, key = lambda s: s.start_sec)
for utt_item in utts_list:
    dialog_rearrange[dialog_name].append(utt_item.utt_name)
f.close

joblib.dump(dialog_rearrange, 'dialog_rearrange.pkl')

total_utt_cnt = 0
for dia_name in dialog_rearrange:
    for utt in dialog_rearrange[dia_name]:
        total_utt_cnt += 1
print(total_utt_cnt)
# check
dialog = joblib.load('dialog.pkl')
for dia in dialog_rearrange:
    if len(dialog_rearrange[dia]) != len(dialog[dia]):
        print(dia, len(dialog_rearrange[dia]), len(dialog[dia]))