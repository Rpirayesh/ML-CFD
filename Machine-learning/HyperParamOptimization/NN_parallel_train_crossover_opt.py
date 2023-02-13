# -*- coding: utf-8 -*-
"""
Created on Sun June 7 16:14:49 2020

@author: JorgeDiaz
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname("FinalOpt/")))
from  OPTFinal import *

#from OptimizedFunction import compile_model
import pickle
import numpy as np
from mpi4py import MPI
import time
import csv
import datetime
import argparse

#generate random integer values
#from random import randrange


def create_csv(file_name, row_list):
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(row_list)


def create_stats(gathered_num_conf, gathered_num_th_accuracy,
                 gathered_num_update_mape, gathered_mape,
                 gathered_std, group_report_size, path, group):
    print("create stats")
    row_list = []
    header = ['group', 'num_conf', 'num_th_accuracy',
              'num_update_mape', 'optimal_mape', 'std_optimal']
    row_list.append(header)
    for gp in range(0, group_report_size):
        row = []
        row.append(gp)
        row.append(gathered_num_conf[gp])
        row.append(gathered_num_th_accuracy[gp])
        row.append(gathered_num_update_mape[gp])
        row.append(gathered_mape[gp])
        row.append(gathered_std[gp])
        row_list.append(row)
    file_name = '%s/group_%s_stats.csv' % (path, group)
    create_csv(file_name, row_list)


def get_config(ModelInfo):
    row = []
    #row.append(conf)
    row.append(ModelInfo['Nerouns'])
    row.append(ModelInfo['Layers'])
    row.append(ModelInfo['Dropout_Value'])

    row.append(ModelInfo['Reguralization'])
    #r = ModelInfo['Reguralization']
    #row.append(r[0].l1)
    #row.append(r[0].l2)

    row.append(ModelInfo['kernel_constraint'])
    #c = ModelInfo['kernel_constraint']
    #row.append(c[0].max_value)

    row.append(ModelInfo['Activation_Method'][0])
    row.append(ModelInfo['Epochs'][0])
    row.append(ModelInfo['Batches'][0])
    row.append(ModelInfo['optimizer'][0])
    w = ModelInfo['W_Initialization_Method']
    w_config = w[0].get_config()
    try:
        d = w_config['distribution']
        d = 'lecun_normal'
    except KeyError:
        d = 'glorot_uniform'
    row.append(d)
    return row


# Make groups of ranks to process the crossover in a parallel way
def make_groups(rank, crossover_size):
    group = rank // crossover_size
    group_comm = MPI.COMM_WORLD.Split(group, rank)
    group_rank = group_comm.Get_rank()
    group_size = group_comm.Get_size()
    print("I am rank ", rank, "and my group is ", group,
          "where I am group_rank", group_rank)
    return group, group_comm, group_rank, group_size


# Make group of ranks to report results
def make_group_report(group_rank):
    group_report_comm = MPI.COMM_WORLD.Split(group_rank, rank)
    group_report_rank = group_report_comm.Get_rank()
    group_report_size = group_report_comm.Get_size()
    print("I am rank ", rank, "and my report group is ", group_rank,
          "where I am group_report_rank", group_report_rank)
    return group_report_comm, group_report_rank, group_report_size


if __name__ == "__main__":
    des = 'Hperparameter search training NNs in parallel with crossover'
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument('-tc', "--total_configurations", default=10, type=int,
                        help='Total number of NNs to train')
    choices = ['energy', 'time', 'error']
    parser.add_argument('-o', "--output", default='energy', choices=choices,
                        help='Output of the NNs')
    parser.add_argument('-obj', "--obj_file", default='EnergyParam.obj',
                        help='Obj file with the configurations to train')
    parser.add_argument('-tq', "--th_quantity", default=10, type=int,
                        help='Threshold quantity')
    parser.add_argument('-ta', "--th_accuracy", default=1, type=float,
                        help='Threshold accuracy')
    parser.add_argument('-cs', "--crossover_size", default=1, type=int,
                        help='Threshold accuracy')
    
    
    args = parser.parse_args()
    total_configurations = args.total_configurations
    output = args.output
    obj_file = args.obj_file
    th_quantity = args.th_quantity
    th_accuracy = args.th_accuracy
    crossover_size = args.crossover_size
 
    # Create global communicator, groups and report group
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    group, group_comm, group_rank, group_size = make_groups(rank, crossover_size)
    grp_rprt_comm, grp_rprt_rank, grp_rprt_size = make_group_report(group_rank)
    
    per_rank = total_configurations//(size//crossover_size)
    filehandler = open(obj_file, 'rb')
    ModelInfo = pickle.load(filehandler)
    
    # print run info
    if rank == 0:
        print('-'*30)
        print("Number of ranks:", size)
        print("Total number of configurations: ", total_configurations)
        print("Configurations per rank: ", per_rank)
        print("Crossover size: ", crossover_size)
        print('-'*30)

    # Create run folder
    path = ''
    if rank == 0:
        now = datetime.datetime.now()
        dt_string = now.strftime("%Y%m%d_%H%M%S")
        path = 'run_%s' % dt_string
        try:
            import os
            os.mkdir(path)
        except:
            pass
    comm.Barrier()

    # Load data and configurations 
    Portion=1
    Feature='Energy'
    InputData,Output,Model=Output_moddel_Data(Portion, Feature)

    print(InputData.head())

    start_time = time.time()
    hyperparam = ['id', 'Nerouns', 'Layers', 'Dropout_Value',
                  'R_l1', 'R_l2',
                  'kernel_constraint', 'Activation_Method',
                  'Epochs', 'Batches', 'optimizer',
                  'W_Initialization_Method']
    row_list = []
    group_row_list = []
    count = 0.0
    mape_temp = 100.0
    std_temp = 0.0
    num_conf = 0
    num_th_accuracy = 0
    num_update_mape = 0
    mape_optimal = np.zeros(2)
    
    z=0

    for conf in range(group*per_rank, (group+1)*per_rank):
        print("I am rank", rank, "running config", conf)
        mapes = []
        num_conf = num_conf + 1
        count = count + 1.0

        evaluated_model, saved_model, mape = compile_model(conf, group_rank, group_size, InputData,Output,Model) # send also group rank and group size and type of output
        
        #mape = randrange(100) + rank
        row = get_config(evaluated_model)
        group_row = get_config(evaluated_model)
        row.append(mape)
        row_list.append(row)
        group_comm.Barrier()
        mapes = group_comm.gather(mape, root=0)
        if group_rank == 0:
            group_row = get_config(evaluated_model)
            average_mape = np.average(mapes)
            std_mape = np.std(mapes)
            group_row.append(average_mape)
            group_row.append(std_mape)
            group_row_list.append(group_row)
            print("I am rank", rank, "and this are the mapes for conf", conf,
                  ":", mapes, "average: ", average_mape, ", std: ", std_mape)
            if mape_temp - average_mape > th_accuracy:
                num_th_accuracy = num_th_accuracy + 1
                count = 0.0
                z = z+1
                th_quantity = 10*z + th_quantity

            if average_mape < mape_temp:
                num_update_mape = num_update_mape + 1
                mape_temp = average_mape
                std_temp = std_mape
                optimal_model_temp = saved_model

        count = group_comm.bcast(count, root=0)
        if count > th_quantity:
            break
    
    if group_rank == 0:
        mape_optimal[0] = mape_temp
        mape_optimal[1] = group
    comm.Barrier()
    
    path = comm.bcast(path, root=0)
    file_name = '%s/%s_output.csv' % (path, rank)
    create_csv(file_name, row_list)
    
    if group_rank == 0:
        file_name = '%s/%s_group_output.csv' % (path, group)
        create_csv(file_name, group_row_list)
    
    gathered_num_conf = grp_rprt_comm.gather(num_conf, root=0)
    gathered_num_th_accuracy = grp_rprt_comm.gather(num_th_accuracy, root=0)
    gathered_num_update_mape = grp_rprt_comm.gather(num_update_mape, root=0)
    gathered_mape_optimal = grp_rprt_comm.gather(mape_optimal[0], root=0)
    gathered_std_optimal = grp_rprt_comm.gather(std_temp, root=0)
    
    
    if rank == 0:
        print("I am rank 0 preparing statistics...")
        create_stats(gathered_num_conf, gathered_num_th_accuracy,
                     gathered_num_update_mape, gathered_mape_optimal,
                     gathered_std_optimal,
                     grp_rprt_size, path, group)
        print("Statistics done")
    
    comm.Barrier()
    
    mape_final = np.zeros(2)
    
    comm.Barrier()
    # Find the minimum MAPE across all ranks
    mape_final = grp_rprt_comm.reduce(mape_optimal, op=MPI.MINLOC, root=0)
    
    comm.Barrier()
    mape_final = comm.bcast(mape_final, root=0)
    
    comm.Barrier()
    #if rank == mape_final[1]:
    #    optimal_model_final = optimal_model_temp
    #    print("I am rank:", mape_final[1], "and I found the optimal model", optimal_model_final)
    #    #print("I am rank:", mape_final[1], "and I found the optimal model")
    
    comm.Barrier()
    if rank == 0:
        print("The optimal mape is: ", mape_final[0],
              "found by group: ", mape_final[1])
        stop_time = time.time()
        total_time = int((stop_time-start_time)*1000)
        print('-'*30)
        print("Total execution time: ", total_time, "ms")
        print('-'*30)
