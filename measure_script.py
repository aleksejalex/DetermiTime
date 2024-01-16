"""
script to measure times (Python)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import sys, os

from matrix_generator import generate_random_matrix, sparsify_mtx, determ_time, generate_square_matrix, \
    make_upper_triangular
from platform_info import get_cpu_info, get_current_datetime

for i in {'a', 'b', 'c'}:
    print(f"char = {i}")  # be careful: it doesn't respect order

for i in ['a', 'b', 'c']:
    print(f"char = {i}")  # now it respects order

set_of_tuples = set()
data = pd.DataFrame(columns=['exec_time', 'dim', 'sparse', 'type', 'n_loops', 'sleep', 'cpu_info', 'curr_time', 'env'])

curr_pc = get_cpu_info()

exp_name = str(input("enter exp. name >>"))

for mtx_dim in [100, 2550, 5000]:
    for mtx_sparsity in [False]:
        for mtx_triangle in [False, True]:
            for mtx_type in [True, False]:
                # curr_matrix = generate_random_matrix(N=mtx_dim, is_integer=mtx_type)
                A = generate_square_matrix(N=mtx_dim, a=0, b=1, steps=mtx_type)
                if mtx_type == True:
                    curr_matrix = np.array(A, dtype=np.float64)
                else:
                    curr_matrix = A
                del A
                if mtx_sparsity == True:
                    curr_matrix = sparsify_mtx(curr_matrix, threshold=0.1)  #threshold=1e-10
                    time.sleep(0.2)  # hard cooldown time for the core working
                if mtx_triangle == True:
                    curr_matrix = make_upper_triangular(curr_matrix)
                for n_loops in [3, 9]:
                    for sleep in [0, 0.15, 0.3]:  # computing core sleeps between every loop of det computation
                        curr_time, _ = determ_time(curr_matrix, ntimes=n_loops, type_of_result='med', sleep_pause=sleep)
                        print(f"Run: dim={mtx_dim}, sparse={mtx_sparsity}, triangle={mtx_triangle}, mtx_type={mtx_type}, "
                              f"comput_time = {n_loops}, RESULT = {curr_time}")

                        row_data = {'exec_time': np.float64(curr_time),
                                    'dim': int(mtx_dim),
                                    'sparse': bool(mtx_sparsity),
                                    'triangle': bool(mtx_triangle),
                                    'type': bool(mtx_type),
                                    'n_loops': int(n_loops),
                                    'sleep': float(sleep),
                                    'curr_time': int(get_current_datetime()),
                                    'env': "script",
                                    'cpu_info': get_cpu_info()}
                        # print(row_data)
                        # data = data.append(row_data, ignore_index=True)  # REMOVED FROM PANDAS!!!
                        data = pd.concat([data, pd.DataFrame([row_data])], ignore_index=False)
                del curr_matrix

# Saving:
save_time = get_current_datetime()
data.to_csv(f'output_{save_time}_{exp_name}.csv', sep=';', index=True)
data.to_pickle(f'output_{save_time}.pkl')
