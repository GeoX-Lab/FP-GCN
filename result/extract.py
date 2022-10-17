from pathlib import Path
import os
from config import config
import re
import csv

outcsv = 'graphfpn.csv'
groupdata = 5
def write_out():
    outpath = os.path.join(config.rootPath, 'result/new_record_log_210311.txt')
    data = Path(outpath).read_text()
    # s = re.compile(r'file:(.\w)\.py', re.L)

    res = re.findall(r'dataset:(\w*)\n', data)
    res1 = re.findall(r'(?:Test Acc:) (.*)\n', data)
    print('res:{},res1:{}'.format(len(res), len(res1)))
    print(res)
    print(res1)
    if 2*len(res) == len(res1):
        with open(outcsv, 'a') as f:
            write = csv.writer(f)
            for i in range(0, len(res), groupdata):
                namedata = [res[i]]
                listdata = []
                for j in range(0+i*2, groupdata*2 +i*2, 2):
                    listdata.append(res1[j])
                namedata.extend(listdata)
                write.writerow(namedata)

write_out()