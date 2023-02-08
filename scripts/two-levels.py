# Initialize dask_mpi
from dask_mpi import initialize
initialize(interface='ib0', dashboard=False)

# Create a Client object
from distributed import Client, get_client
client = Client()

import pandas as pd
import numpy as np
from math import isclose

import time
import random

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


def subtask(chunk, chunk_name):
    print(f"@@@ starting subtask {chunk_name} ...")
    result = sum(chunk)
    print(f"@@@ returning sum of {chunk_name} = {result}")
    return result


def task(series: pd.Series, column_name: str) -> float:
    """Compute the sum of the dataframe column <series>.
    
    :param series: a dataframe column (numeric)
    :param column_name: name of the column in the dataframe, to produce a intelligible message in the output. 
    """
    # pretend the task takes some time
    print(f"@@  starting task '{column_name}' ...")
    client = get_client()
    n_workers = 5
    chunk_names = []
    futures = []
    for i in range(n_workers):
        i0 = i * 10
        i1 = i0 + 10
        chunk = series[i0:i1]
        chunk_name = f"{column_name}[{i0}:{i1}]"
        chunk_names.append(chunk_name)
        print(f"@@  creating future for subtask '{chunk_name}' ...")
        future = client.submit(subtask, chunk, chunk_name)
        futures.append(future)
    
    results = client.gather(futures)
    result = sum(results)
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
