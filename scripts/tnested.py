# According to http://mpi.dask.org/en/latest/batch.html

from dask_mpi import initialize
initialize(interface='ib0', dashboard=False)
print("initialized")

from distributed import Client
client = Client()
print(f"{client=}")

import time
import random
from mpi4py import MPI
from mpi_print import print
from math import isclose

import os 
filename = f"{os.environ['PBS_JOBNAME']}.{os.environ['PBS_JOBID']}.output.txt"
# because there is a lot of I/O from the workers and the scheduler to stdout/stderr, 
# we prefer to write the output of our application to a separate file.


with open(filename, mode='a') as f:
    print('rank starting', file=f)


def less_costly_simulation(chunk, chunk_name):
    """Further subdivide the work over n_workers workers"""
    time.sleep(random.random())    
    result = sum(chunk)
    with open(filename, mode='a') as f:        
        print(f"sum of chunk '{chunk_name}' = ({result})", file=f)
        
    return result

def costly_simulation(column, column_name, n_workers):
    """Further subdivide the work over n_workers workers"""
    futures = []
    
    column_size = len(column)
    chunk_size = int(column_size / n_workers)
    for i in range(n_workers):
        i0 = chunk_size * 1
        i1 = min(i0 + chunk_size, column_size)
        future = client.submit(less_costly_simulation, column[i0:i1], f"{column_name}[{i0}:{i1}]")
        futures.append(future)
    
    results = []
    for future in futures:
        results.append(future.result())
        
    result = sum(results)
    
    with open(filename, mode='a') as f:        
        print(f"sum of column '{column_name}' = ({result})", file=f)
    
    return result

import pandas as pd
import numpy as np
from et_stopwatch import Stopwatch

if __name__ == "__main__":
    # This code runs only on the client rank (1)
    # (Rank 0 runs the scheduler and all other ranks are workers)
      
    # Create a numpy n_rows x n_columns matrix of random numbers. 
    n_columns = 100
    n_rows = 50
    a = np.random.random(size=(n_rows, n_columns))
    # Scale the column such that the sum of the i-th column is i+1
    # The total sum is then 1+2+...+n_columns = n_columns*(n_columns+1)/2
    c = sum(a)
    for i in range(len(c)):
        c[i] /= i+1
    a /= c
    # now the sum of the 
    if not isclose(sum(sum(a)), n_columns*(n_columns+1)/2, rel_tol=1.e-6):
        raise ValueError('initialization of a is not ok.')
    
    # create a dataframe from the matrix a: 
    column_names = [f"column_{c}" for c in range(n_columns)] 
    df = pd.DataFrame(a, columns=column_names)

    with Stopwatch("This took") as sw:
        # We will have the workers (2-127) compute the sum of the individual 
        # columns and finally accumulate these sums in the client rank (1).
        # The difference between dask futures and concurrent.futures is that 
        # using dask we can engage many nodes as we want to distribute the work.
        
        futures = []
        for column_name in df.columns:
            future = client.submit(costly_simulation, df[column_name], column_name, n_workers=5)
            futures.append(future)
            
        results = []
        for future in futures:
            result = future.result()
            with open(filename, mode='a') as f:
                print(f"{result=}", file=f)
            results.append(result)
        
        result = sum(results)
        
    with open(filename, mode='a') as f:
        print(f"sum {result=}", file=f)
