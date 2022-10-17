import os
import sys
import time

base_path = os.getcwd()
father_path = os.path.abspath("../result")
print(base_path)
print(os.path.join(base_path,"data"))
print(father_path)
logfile = os.path.join(father_path, 'new_record_log_{}.txt'.format(int(time.time())))
print(logfile)


rootPath = os.path.abspath(os.path.dirname(__file__))
print(rootPath)
path22 = os.path.abspath(os.path.join(rootPath, "../"))
print(os.path.join(path22, "results"))