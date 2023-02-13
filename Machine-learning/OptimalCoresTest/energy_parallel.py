import numpy
from mpi4py import MPI
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

total_configurations = 10

per_rank = total_configurations//size

# print important info
if rank == 0:
    print('-'*30)
    print("Number of ranks:", size)
    print("Total number of configurations: ", total_configurations)
    print("Configurations per rank: ", per_rank)
    print('-'*30)

comm.Barrier()
start_time = time.time()

for conf in range(1 + rank*per_rank, 1 + (rank+1)*per_rank):
    print("I am rank ", rank, " running conf ", conf)

comm.Barrier()

if rank == 0:
    # Process remaining configurations
    for conf in range(1 + (size)*per_rank, total_configurations+1):
        print("I am rank ", rank, " running conf ", conf)
    stop_time = time.time()
    total_time = int((stop_time-start_time)*1000)
    print('-'*30)
    print("time spent with ", size, " ranks: ", total_time, "ms")
    print('-'*30)
