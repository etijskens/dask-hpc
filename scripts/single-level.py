import os
from math import isclose
import time

from mpi4py import MPI

try:
    on_vsc = os.environ['VSC_INSTITUTE_CLUSTER'] != 'local'
except KeyError:
    on_vsc = False
print(f'{on_vsc=}')
mpi_comm = MPI.COMM_WORLD
mpi_rank = mpi_comm.Get_rank()
mpi_size = mpi_comm.Get_size()

if mpi_size > 1:
    # Initialize dask_mpi
    from dask_mpi import initialize
    interface = 'ib0' if on_vsc else None
    if not interface:
        initialize(dashboard=False)
    else:
        initialize(interface=interface, dashboard=False)

    # Create a Client object
    from distributed import Client
    client = Client()

import pandas as pd
import numpy as np


print(f'@   rank 1 starting client script')
#   There is a lot of output from dask to stdout. To distinghuish we let every line
#   written by our script start with "@   " (client script) or "@@  " (workers).

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


def task(series: pd.Series, column_name: str) -> float:
    """Compute the sum of the dataframe column <series>.
    
    :param series: a dataframe column (numeric)
    :param column_name: name of the column in the dataframe, to produce a intelligible message in the output. 
    """
    # pretend the task takes some time
    print(f"@@  starting task '{column_name}' ...")
    time.sleep(np.random.uniform())
    result = sum(series)
    print(f"@@  Returning sum of '{column_name}' = ({result:8.3f})")
    return result


if __name__ == "__main__":
    df = create_dataframe(n_columns=10, n_rows=50) 

    ### Distribute the work ###  
    
    futures = []
    for column_name in df.columns:
        print(f"@   creating future for task '{column_name}' ...")
        future = client.submit(task, df[column_name], column_name)
        futures.append(future)

    print("@   Retrieving task results ...")
    results = client.gather(futures)
    lines = []
    for r in range(len(results)):
        lines.append(f"@   Partial sum {df.columns[r]:<10} = {results[r]:8.3f}")
    lines.append(    f"@     Total sum            = {sum(results):8.3f}")
    lines = "\n".join(lines)
    print(lines)
    print('-*# finished #*-')