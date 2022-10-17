import os
import time
# os.system('source ./data/env/geo_toorch/bin/activate')
from config import config

outname = config.dataname + '.txt'
import sys
sys.stdout = open(outname, 'wt')

path = os.getcwd()

files = os.listdir(path)

files = ['sort_pool.py']
for file in files:
    if os.path.splitext(file)[1]=='.py' and file!='batch_program.py':
        start = time.time()
        for i in range(100, 125, 5):
            print('file:%s, seed: %d' %(file, i))
            # os.system('python %s --seed %d' % (file, i))
            out = os.popen('python %s --seed %d' % (file, i))
            # out.read()
            print(out.read())
        print(file, 'used time is :', time.time()-start)
print('sucess!')

