import os
import sys

class pro():
    rootPath = os.path.abspath(os.path.dirname(__file__))
    sys.path.insert(0, rootPath)
    logfile = os.path.join(rootPath, 'new_record_log.txt')
    epoch = 1000
    batch_size = 128
    data_split = 3

    dataname = 'DD'
    # dataname = 'PROTEINS'
    # dataname = 'NCI1'
    # dataname = 'NCI109'
    # dataname = 'FRANKENSTEIN'


    # dataname = 'MUTAG'
    # dataname = 'PROTEINS'
    # dataname = 'DD'
    # dataname = 'MSRC_21'
    # dataname = 'DHFR'




    #delect
    # dataname = 'ENZYMES'
    #没有节点特征, use_node_attr=True
    # dataname = 'Synthie'
    # dataname = 'PTC_FM'
    # dataname = 'PTC_FR'
    # dataname = 'PTC_MM'
    # dataname = 'PTC_MR'
config = pro()
