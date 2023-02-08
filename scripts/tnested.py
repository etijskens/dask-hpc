# According to http://mpi.dask.org/en/latest/batch.html

from dask_mpi import initialize
initialize(interface='ib0', dashboard=False)
print("initialized")

from distributed import Client, get_client
client = Client()

import time
import random
from mpi4py import MPI
from math import isclose
import pandas as pd
import numpy as np

# from mpi_print import print
from et_stopwatch import Stopwatch
import et_stopwatch
et_stopwatch.print = print # use mpi_print's print inside et_stopwatch


# Pick a filename for the log file: '$PBS_JOBNAME.$PBS_JOBID.output.txt'
# (because there is a lot of I/O from the workers and the scheduler to stdout/stderr, 
# we prefer to write the output of our application to a separate log file).
import os 
filename = f"{os.environ['PBS_JOBNAME']}.{os.environ['PBS_JOBID']}.output.txt"

# with open(filename, mode='a') as f:
#     print('Client script starting ...', file=f)


def less_costly_simulation(chunk, chunk_name, sleep):
    """compute sum of a chunk"""
    # from mpi_print import print
    # import et_stopwatch
    # et_stopwatch.print = print # use mpi_print's print inside et_stopwatch
    
    # with et_stopwatch.Stopwatch(f"less_costly_simulation(chunk, {chunk_name=}, {sleep=}) took:"):
    print(f"@@@ subtask starting for {chunk_name}")
    # time.sleep(random.random())
    result = sum(chunk)
        # with open(filename, mode='a') as f:        
        #     print(f"sum of '{chunk_name}' = ({result})", file=f)
    print(f"@@@ returning sum({chunk_name}) = {result}")
        
    return result

def costly_simulation(series: pd.Series, column_name: str, n_workers: int) -> float:
    """Further subdivide the work over n_workers workers."""
    # from mpi_print import print
    from et_stopwatch import Stopwatch
    et_stopwatch.print = print # use mpi_print's print inside et_stopwatch
    
    client = get_client()
    futures = []
    chunk_names = []
    # with Stopwatch(f"Creating chunk futures for column {column_name} took:", file=filename):
    column_size = len(series)
    chunk_size = int(column_size / n_workers)
    sleep = column_size/n_workers
    for i in range(n_workers):
        i0 = chunk_size * i
        i1 = min(i0 + chunk_size, column_size)
        chunk_name = f"{column_name}[{i0}:{i1}]"
        print("@@  creating subtask future for", chunk_name)
        future = client.submit(less_costly_simulation, series[i0:i1], chunk_name, sleep)
        futures.append(future)
        chunk_names.append(chunk_name)
        
    # with Stopwatch(f"Retrieving chunk results for column {column_name} took:", file=filename):
    print(f"`@@  gathering subtasks for {column_name}")
    results = client.gather(futures)
    result = sum(results)
    # do some I/O
    # lines = []
    # for r in range(len(results)):
    #     lines.append(f"result {r} {chunk_names[r]} result={results[r]}")
    # lines.append(f"{column_name} sum = {result}")
    # lines = '\n'.join(lines)
    # with open(filename, mode='a') as f:
    #     print(lines, file=f)
            
    return result


def create_dataframe(*, n_rows: int, n_columns:int) -> pd.DataFrame:
    """ Create a DataFrame with n_rows x n_columns, filled with random floats between 0 and 1. 
    Columns are scaled such that their sum equals i+1, i being the column index (starting counting from 0).
    """
    # Create a numpy n_rows x n_columns matrix of random floats. 
    n_columns = 10
    n_rows = 50
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
    
    # with Stopwatch("Creating futures took:", file=filename):
    futures = []
    for column_name in df.columns[0:9]:
        print(f"@  Creating task future for {column_name}")
        future = client.submit(costly_simulation, df[column_name], column_name, 5)
        futures.append(future)

    # with Stopwatch("Retrieving column results took:", file=filename):
    print("@   gathering tasks")
    results = client.gather(futures)
    lines = []
    for r in range(len(results)):
        lines.append(f"@   Partial sum {df.columns[r]} = {results[r]:8.3f}")
    lines.append(    f"@     Total sum          = {sum(results):8.3f}")
    lines = "\n".join(lines)
    print(lines)
