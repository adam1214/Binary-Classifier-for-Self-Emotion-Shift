import numpy as np
import argparse
from argparse import RawTextHelpFormatter

parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
parser.add_argument('-r', "--file_name", type=int, help="select file", default='')
args = parser.parse_args()

f = open(args.file_name)
num_list = []
for line in f.readlines():
    #print(float(line))
    num_list.append(float(line))

arr = np.asarray(num_list)

print(arr.shape)
print(np.std(arr, ddof=1))
print(np.mean(arr))
f.close()