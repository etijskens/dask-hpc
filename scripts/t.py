# According to http://mpi.dask.org/en/latest/batch.html

from dask_mpi import initialize
initialize(interface='ib0', dashboard=False)
print("initialized")

from distributed import Client
client = Client()
print(f"{client=}")

import pandas as pd
import numpy as np
from math import isclose

import time
import random
# Fix the seed so that we get reproducible runs:
random.seed(123456) # any number will do.
# (OS jitter may cause additional variation in timings) 
# Hm, that does not go as expected. Apparently the line above is not executed on the 
# workers, so, they have a ramdom seed. 
# We have to pass the random value from the client to the workers. 

from et_stopwatch import Stopwatch
from mpi_print import print
# make Stopwatch use mpi_print.print.
import et_stopwatch
et_stopwatch.print = print 

# Pick a filename for the log file: '$PBS_JOBNAME.$PBS_JOBID.output.txt'
# (because there is a lot of I/O from the workers and the scheduler to stdout/stderr, 
# we prefer to write the output of our application to a separate log file).
import os 
filename = f"{os.environ['PBS_JOBNAME']}.{os.environ['PBS_JOBID']}.output.txt"

# write a message that the client script started:
with open(filename, mode='a') as f:
    print('rank starting client script', file=f)


def costly_simulation(series: pd.Series, column_name: str, sleep: float) -> float:
    """Compute the sum of the dataframe column <series>.
    
    :param series: a dataframe column (numeric)
    :param column_name: name of the column in the dataframe, to produce a intelligible message in the log file. 
    :param sleep: the number of seconds this function will sleep (to simulate a long compute time). Originally
        we used random.random() for that, but we cannot set the seed of the random generator in the workers. 
        Hence, we pass the value from the client script. 
    """
    from mpi_print import print
    et_stopwatch.print = print
    
    with Stopwatch(f"costly_simulation(column_data, {column_name=}, {sleep=}) took:", file=filename):
        time.sleep(sleep)
        result = sum(series)
        with open(filename, mode='a') as f:
            print(f"sum of '{column_name}' = ({result:8.3f})", file=f)

    return result


def create_dataframe(*, n_rows: int, n_columns:int) -> pd.DataFrame:
    """ Create a DataFrame with n_rows x n_columns, filled with random floats between 0 and 1. 
    Columns are scaled such that their sum equals i+1, i being the column index (starting counting from 0).
    """
    # Create a numpy n_rows x n_columns matrix of random floats. 
    a = np.random.random(size=(n_rows, n_columns))
    # Scale the column such that the sum of the i-th column is i+1 (but it is named 'column_<i+1>)
    # The total sum is then 1+2+...+n_columns = n_columns*(n_columns+1)/2
    c = sum(a)
    for i in range(len(c)):
        c[i] /= i+1
    a /= c
    # now the sum of the 
    if not isclose(sum(sum(a)), n_columns*(n_columns+1)/2, rel_tol=1.e-6):
        raise ValueError('initialization of a is not ok.')
    
    # create a dataframe from the matrix a: 
    column_names = [f"column_{c+1}" for c in range(n_columns)] # column_names
    
    return pd.DataFrame(a, columns=column_names)


if __name__ == "__main__":
    # This code runs only on the client rank (1)
    # (Rank 0 runs the scheduler and all other ranks are workers)
    
    df = create_dataframe(n_columns=10, n_rows=50) 

    ### Distribute the work ###  
    # We will compute the sum of all elements in the dataframe df. 
    # We will use the workers (all ranks > 1) to compute the sum of the individual 
    # columns and finally accumulate these partial sums in the client rank (1).
    # We use dask futures for this. The difference between dask futures and 
    # concurrent.futures is that using dask we can engage as many nodes as we 
    # want to distribute the work.
    
    with Stopwatch("Creating futures took:", file=filename):
        futures = []
        for column_name in df.columns:
            future = client.submit(costly_simulation, df[column_name], column_name, random.random())
            futures.append(future)

    with Stopwatch("Retrieving column results took:", file=filename):
        results = client.gather(futures)
        lines = []
        for r in range(len(results)):
            lines.append(f"Partial sum {df.columns[r]:<10} = {results[r]:8.3f}")
        lines.append(    f"Total   sum            = {sum(results):8.3f}")
        lines = "\n".join(lines)
        with open(filename, mode='a') as f:
            print(lines, file=f)
