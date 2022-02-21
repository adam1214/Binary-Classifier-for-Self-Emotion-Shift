import joblib
f = open('AllTranscriptions.txt', 'r')
utt_time_dict = {}
for line in f.readlines():
    line_list = line.split()
    try:
        if line_list[1][0] == '[' and line_list[1][-1] == ':':
            time_str = line_list[1]
            print(time_str, line_list[0])
            utt_time_dict[line_list[0]] = {}
            utt_time_dict[line_list[0]]['start_sec'] = float(time_str[1:9])
            utt_time_dict[line_list[0]]['end_sec'] = float(time_str[10:18])
    except:
        continue
f.close
joblib.dump(utt_time_dict, 'utt_time_dict.pkl')
