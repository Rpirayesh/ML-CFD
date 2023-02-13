# -*- coding: utf-8 -*-
"""
Created on Fri May 29 09:03:49 2020

@author: rpira
@author: JorgeDiaz
"""

from OptimizedFunction import compile_model
import pickle
import numpy as np
from mpi4py import MPI
import time
import csv
import os
import datetime
import argparse

#generate random integer values
#from random import randrange


def create_csv(file_name, row_list):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def create_stats(gathered_num_conf, gathered_num_th_accuracy, gathered_num_update_mape, gathered_mape, size, path):
    row_list = []
    header = ['rank', 'num_conf', 'num_th_accuracy', 'num_update_mape', 'optimal_mape']
    row_list.append(header)
    for rank in range(0, size):
        row = []
        row.append(rank)
        row.append(gathered_num_conf[rank])
        row.append(gathered_num_th_accuracy[rank])
        row.append(gathered_num_update_mape[rank])
        row.append(gathered_mape[rank])
        row_list.append(row)
    file_name = '%s/stats.csv' %path
    create_csv(file_name, row_list)


def get_config(conf):
    row = []
    row.append(conf)
    row.append(ModelInfo[conf]['Nerouns'])
    row.append(ModelInfo[conf]['Layers'])
    row.append(ModelInfo[conf]['Dropout_Value'])
    r = ModelInfo[conf]['Reguralization']
    row.append(r[0].l1)
    row.append(r[0].l2)
    c = ModelInfo[conf]['kernel_constraint']
    row.append(c[0].max_value)
    row.append(ModelInfo[conf]['Activation_Method'][0])
    row.append(ModelInfo[conf]['Epochs'][0])
    row.append(ModelInfo[conf]['Batches'][0])
    row.append(ModelInfo[conf]['optimizer'][0])
    w = ModelInfo[conf]['W_Initialization_Method']
    w_config = w[0].get_config()
    try:
        d = w_config['distribution']
        d = 'lecun_normal'
    except KeyError:
        d = 'glorot_uniform'
    row.append(d)
    return row


parser = argparse.ArgumentParser(description='Hperparameter search training NNs in parallel')
parser.add_argument('-tc', "--total_configurations", default=10, type=int,
                    help='Total number of NNs to train')
parser.add_argument('-o', "--output", default='energy', choices=['energy', 'time', 'error'],
                    help='Output of the NNs')
parser.add_argument('-obj', "--obj_file", default='filename_pi_10.obj',
                    help='Obj file with the configurations to train')
parser.add_argument('-tq', "--th_quantity", default=10, type=int,
                    help='Threshold quantity')
parser.add_argument('-ta', "--th_accuracy", default=1, type=int,
                    help='Threshold accuracy')


args = parser.parse_args()
total_configurations = args.total_configurations
output = args.output
obj_file = args.obj_file
th_quantity = args.th_quantity
th_accuracy = args.th_accuracy

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

per_rank = total_configurations//size
filehandler = open(obj_file, 'rb')
ModelInfo = pickle.load(filehandler)

# print run info
if rank == 0:
    print('-'*30)
    print("Number of ranks:", size)
    print("Total number of configurations: ", total_configurations)
    print("Configurations per rank: ", per_rank)
    print('-'*30)

mape_optimal = np.zeros(2)

comm.Barrier()
start_time = time.time()

hyperparam = ['id', 'Nerouns', 'Layers', 'Dropout_Value',\
              'R_l1', 'R_l2',\
              'kernel_constraint', 'Activation_Method',\
              'Epochs', 'Batches', 'optimizer',\
              'W_Initialization_Method']
row_list = []
count = 0.0
mape_temp = 100.0
num_conf = 0
num_th_accuracy = 0
num_update_mape = 0
for conf in range(rank*per_rank, (rank+1)*per_rank):
    num_conf = num_conf + 1
    count = count + 1.0
    row = get_config(conf)
    saved_model, mape = compile_model(ModelInfo[conf])
    #mape = randrange(100)
    row.append(mape)
    row_list.append(row)
    print("I am rank", rank, "running conf", conf, ". MAPE =", mape)
    if mape_temp - mape > th_accuracy:
        num_th_accuracy = num_th_accuracy + 1
        count = 0.0
    if mape < mape_temp:
        num_update_mape = num_update_mape + 1
        mape_temp = mape
        optimal_model_temp = saved_model
    if count > th_quantity:
        break

mape_optimal[0] = mape_temp
mape_optimal[1] = rank

comm.Barrier()

path = ''

if rank == 0:
    # Process remaining configurations
    count = 0.0
    for conf in range(size*per_rank, total_configurations):
        num_conf = num_conf + 1
        count = count + 1.0
        row = get_config(conf)
        saved_model, mape  =compile_model(ModelInfo[conf])
        #mape = randrange(100)
        row.append(mape)
        row_list.append(row)
        print("I am rank", rank, "running conf", conf, ". MAPE =", mape)
        if mape_temp - mape > th_accuracy:
            num_th_accuracy = num_th_accuracy + 1
            count = 0.0
        if mape < mape_temp:
            num_update_mape = num_update_mape + 1
            mape_temp = mape
            optimal_model_temp = saved_model
        if count > th_quantity:
            break
    mape_optimal[0] = mape_temp

    now = datetime.datetime.now()
    dt_string = now.strftime("%Y%m%d_%H%M%S")
    path = 'run_%s' %dt_string
    try:
        os.mkdir(path)
    except:
        pass

comm.Barrier()
path = comm.bcast(path, root=0)   
file_name = '%s/%s_output.csv' %(path, rank)
create_csv(file_name, row_list)

gathered_row_list = comm.gather(row_list, root=0)
gathered_num_conf = comm.gather(num_conf, root=0)
gathered_num_th_accuracy = comm.gather(num_th_accuracy, root=0)
gathered_num_update_mape = comm.gather(num_update_mape, root=0)
gathered_mape_optimal = comm.gather(mape_optimal[0], root=0)

if rank == 0:
    final_results = []
    final_results.append(hyperparam)
    final_results.append(gathered_row_list)
    file_name = '%s/total_output.csv' %path
    create_csv(file_name, final_results)
    create_stats(gathered_num_conf, gathered_num_th_accuracy, gathered_num_update_mape, gathered_mape_optimal, size, path)

mape_final = np.zeros(2)

comm.Barrier()
# Find the minimum MAPE across all ranks
mape_final = comm.reduce(mape_optimal, op=MPI.MINLOC, root=0)

comm.Barrier()
mape_final = comm.bcast(mape_final, root=0)

comm.Barrier()
if rank == mape_final[1]:
    optimal_model_final = optimal_model_temp
    print("I am rank:", mape_final[1], "and I found the optimal model", optimal_model_final)
    #print("I am rank:", mape_final[1], "and I found the optimal model")

comm.Barrier()
if rank == 0: 
    print("The optimal mape is: ", mape_final[0], "found by rank: ", mape_final[1])

    stop_time = time.time()
    total_time = int((stop_time-start_time)*1000)
    print('-'*30)
    print("Total execution time: ", total_time, "ms")
    print('-'*30)
