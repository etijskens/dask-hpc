# Report on dask 

Dask futures are used to distribute a problem into tasks (e.g. the replicas) and subtasks (e.g. the species loop in a replica).
On the cluster we use dask-mpi which autmatically sets up a scheduler on rank 0, the client script on rank 1, and a number of 
workers on the remaining ranks.

## The toy problem

The toy problem treated here is to compute the sum of a 5x10 matrix of random numbers. The colums of the matrix are scaled such
that the sum of the *i*-th colum (i = 1..10) equals *i*. 

Two scenarios were envisaged for the toy problem:

1. one level of tasks. One task (future) is created for each column. So, each task computes a column sum. 
2. two task levels. Each column sum is split in 5 subtasks computing the sum of elements 0:10 of the column, elements 10:20, ...
   
## One task level

This is the script: `single-level.py`. The corresponding job script is single-level.pbs. The job script requests 12 cores, 
which implies 10 workers, as rank 0 and 1 are used for the dask scheduler and the client script, respectively. 

```python
from dask_mpi import initialize # Initialize dask_mpi
initialize(interface='ib0', dashboard=False)
from distributed import Client
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


def task(series: pd.Series, column_name: str) -> float:
    """Compute the sum of the dataframe column <series>.
    
    :param series: a dataframe column (numeric)
    :param column_name: name of the column in the dataframe, to produce a intelligible message in the output. 
    """
    # pretend the task takes some time
    print(f"@@  starting task '{column_name}' ...")
    time.sleep(random.random())
    result = sum(series)
    print(f"@@  Returning sum of '{column_name}' = ({result:8.3f})")
    return result


if __name__ == "__main__":
    df = create_dataframe(n_columns=10, n_rows=50) 

    client = Client() # dask Client object

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
```

Here is the output (with the output from dask removed:)
```
@   rank 1 starting client script
@    creating future for task 'column_1' ...
@    creating future for task 'column_2' ...
@    creating future for task 'column_3' ...
@    creating future for task 'column_4' ...
@    creating future for task 'column_5' ...
@@  starting task 'column_1' ...
@    creating future for task 'column_6' ...
@    creating future for task 'column_7' ...
@    creating future for task 'column_8' ...
@    creating future for task 'column_9' ...
@    creating future for task 'column_10' ...
@   Retrieving task results ...
@@  starting task 'column_4' ...
@@  starting task 'column_2' ...
@@  starting task 'column_7' ...
@@  starting task 'column_8' ...
@@  starting task 'column_6' ...
@@  starting task 'column_3' ...
@@  starting task 'column_5' ...
@@  starting task 'column_9' ...
@@  starting task 'column_10' ...
@@  Returning sum of 'column_6' = (   6.000)
@@  Returning sum of 'column_2' = (   2.000)
@@  Returning sum of 'column_1' = (   1.000)
@@  Returning sum of 'column_4' = (   4.000)
@@  Returning sum of 'column_5' = (   5.000)
@@  Returning sum of 'column_9' = (   9.000)
@@  Returning sum of 'column_10' = (  10.000)
@@  Returning sum of 'column_8' = (   8.000)
@@  Returning sum of 'column_7' = (   7.000)
@@  Returning sum of 'column_3' = (   3.000)
@   Partial sum column_1   =    1.000
@   Partial sum column_2   =    2.000
@   Partial sum column_3   =    3.000
@   Partial sum column_4   =    4.000
@   Partial sum column_5   =    5.000
@   Partial sum column_6   =    6.000
@   Partial sum column_7   =    7.000
@   Partial sum column_8   =    8.000
@   Partial sum column_9   =    9.000
@   Partial sum column_10  =   10.000
@     Total sum            =   55.000
```

## Two task levels

The python script is `two-levels.py`. The corresponding job script `two-levels.pbs` requests 20 cores. It was found that when 
the number of tasks equals the number of workers, the subtasks do not start and the program terminates when the wall time limit
is reached without doing any useful work. This is probably also the case when there are more taks tasks than workers.

Also, note thet in the program below the tasks distribute their work over the subtasks and then wait for the subtasks to finish.
It would probably be more efficient if the tasks would take on one subtask themselves once the subtask futures are created. In the current script, the 10 cores running a task are, in fact, wasted, as is the core running the client script.

This is the python script:
```python

from dask_mpi import initialize
initialize(interface='ib0', dashboard=False) # Initialize dask_mpi
from distributed import Client, get_client
import pandas as pd
import numpy as np
from math import isclose
import time
import random

print(f'@   rank 1 starting client script')
#   There is a lot of output from dask to stdout. To distinghuish we let every line
#   written by our script start with "@   " (client script) or "@@  " (tasks), and "@@@ subtasks".

def create_dataframe(*, n_rows: int, n_columns:int) -> pd.DataFrame:
    """ as before """


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
    client = get_client() # get hold of the client object to distribute the work.
    n_subtasks = 5 
    futures = []
    for i in range(n_subtasks):
        i0 = i * 10
        i1 = i0 + 10
        chunk = series[i0:i1]
        chunk_name = f"{column_name}[{i0}:{i1}]"
        print(f"@@  creating future for subtask '{chunk_name}' ...")
        future = client.submit(subtask, chunk, chunk_name)
        futures.append(future)
    
    results = client.gather(futures)
    result = sum(results)
    print(f"@@  Returning sum of '{column_name}' = ({result:8.3f})")
    return result


if __name__ == "__main__":
    df = create_dataframe(n_columns=10, n_rows=50) 
    client = Client() # create a dask Client object
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

```

And her is its (rather long) output.
```
@   rank 1 starting client script
@   creating future for task 'column_1' ...
@   creating future for task 'column_2' ...
@   creating future for task 'column_3' ...
@   creating future for task 'column_4' ...
@   creating future for task 'column_5' ...
@@  starting task 'column_1' ...
@   creating future for task 'column_6' ...
@   creating future for task 'column_7' ...
@   creating future for task 'column_8' ...
@   creating future for task 'column_9' ...
@   creating future for task 'column_10' ...
@   Retrieving task results ...
@@  starting task 'column_2' ...
@@  starting task 'column_5' ...
@@  starting task 'column_3' ...
@@  starting task 'column_6' ...
@@  starting task 'column_4' ...
@@  starting task 'column_10' ...
@@  starting task 'column_9' ...
@@  starting task 'column_8' ...
@@  starting task 'column_7' ...
@@  creating future for subtask 'column_5[0:10]' ...
@@  creating future for subtask 'column_7[0:10]' ...
@@  creating future for subtask 'column_6[0:10]' ...
@@  creating future for subtask 'column_8[0:10]' ...
@@  creating future for subtask 'column_2[0:10]' ...
@@  creating future for subtask 'column_10[0:10]' ...
@@  creating future for subtask 'column_9[0:10]' ...
@@  creating future for subtask 'column_3[0:10]' ...
@@  creating future for subtask 'column_4[0:10]' ...
@@  creating future for subtask 'column_6[10:20]' ...
@@  creating future for subtask 'column_5[10:20]' ...
@@  creating future for subtask 'column_7[10:20]' ...
@@  creating future for subtask 'column_5[20:30]' ...
@@  creating future for subtask 'column_6[20:30]' ...
@@@ starting subtask column_6[0:10] ...
@@@ returning sum of column_6[0:10] = 1.0954556297881017
@@  creating future for subtask 'column_1[10:20]' ...
@@  creating future for subtask 'column_5[30:40]' ...
@@  creating future for subtask 'column_7[20:30]' ...
@@  creating future for subtask 'column_6[30:40]' ...
@@@ starting subtask column_1[0:10] ...
@@@ returning sum of column_1[0:10] = 0.2322288863842889
@@  creating future for subtask 'column_5[40:50]' ...
@@@ starting subtask column_7[0:10] ...
@@@ returning sum of column_7[0:10] = 1.1942569816997215
@@  creating future for subtask 'column_1[20:30]' ...
@@  creating future for subtask 'column_7[30:40]' ...
@@  creating future for subtask 'column_6[40:50]' ...
@@  creating future for subtask 'column_8[10:20]' ...
@@  creating future for subtask 'column_1[30:40]' ...
@@  creating future for subtask 'column_7[40:50]' ...
@@  creating future for subtask 'column_2[10:20]' ...
@@  creating future for subtask 'column_8[20:30]' ...
@@  creating future for subtask 'column_1[40:50]' ...
@@  creating future for subtask 'column_9[10:20]' ...
@@@ starting subtask column_8[0:10] ...
@@@ returning sum of column_8[0:10] = 1.5517904199643255
@@  creating future for subtask 'column_10[10:20]' ...
@@  creating future for subtask 'column_8[30:40]' ...
@@  creating future for subtask 'column_2[20:30]' ...
@@@ starting subtask column_2[0:10] ...
@@@ returning sum of column_2[0:10] = 0.43416120815640513
@@  creating future for subtask 'column_9[20:30]' ...
@@  creating future for subtask 'column_10[20:30]' ...
@@  creating future for subtask 'column_2[30:40]' ...
@@  creating future for subtask 'column_8[40:50]' ...
@@@ starting subtask column_10[0:10] ...
@@@ returning sum of column_10[0:10] = 2.421218264483018
@@  creating future for subtask 'column_9[30:40]' ...
@@  creating future for subtask 'column_10[30:40]' ...
@@  creating future for subtask 'column_2[40:50]' ...
@@  creating future for subtask 'column_9[40:50]' ...
@@  creating future for subtask 'column_3[10:20]' ...
@@  creating future for subtask 'column_10[40:50]' ...
@@  creating future for subtask 'column_4[10:20]' ...
@@  creating future for subtask 'column_3[20:30]' ...
@@  creating future for subtask 'column_4[20:30]' ...
@@@ starting subtask column_4[0:10] ...
@@@ returning sum of column_4[0:10] = 0.8020783959483536
@@  creating future for subtask 'column_3[30:40]' ...
@@  creating future for subtask 'column_4[30:40]' ...
@@  creating future for subtask 'column_3[40:50]' ...
@@  creating future for subtask 'column_4[40:50]' ...
@@@ starting subtask column_3[0:10] ...
@@@ returning sum of column_3[0:10] = 0.6496971282662344
@@@ starting subtask column_9[0:10] ...
@@@ returning sum of column_9[0:10] = 1.798074127627648
@@@ starting subtask column_8[40:50] ...
@@@ returning sum of column_8[40:50] = 1.470048006758271
@@@ starting subtask column_8[30:40] ...
@@@ returning sum of column_8[30:40] = 1.783277650271304
@@@ starting subtask column_8[10:20] ...
@@@ returning sum of column_8[10:20] = 1.6768449700188175
@@@ starting subtask column_8[20:30] ...
@@@ returning sum of column_8[20:30] = 1.5180389529872795
@@@ starting subtask column_5[0:10] ...
@@@ returning sum of column_5[0:10] = 1.0663319204283381
@@@ starting subtask column_7[10:20] ...
@@@ returning sum of column_7[10:20] = 1.2782199800144651
@@@ starting subtask column_7[20:30] ...
@@@ returning sum of column_7[20:30] = 1.6111975455025052
@@@ starting subtask column_3[20:30] ...
@@@ returning sum of column_3[20:30] = 0.5513020379447751
@@@ starting subtask column_3[30:40] ...
@@@ returning sum of column_3[30:40] = 0.6031127691638102
@@@ starting subtask column_3[40:50] ...
@@@ returning sum of column_3[40:50] = 0.5343457677877534
@@@ starting subtask column_10[20:30] ...
@@@ returning sum of column_10[20:30] = 2.829769588815964
@@@ starting subtask column_10[30:40] ...
@@@ returning sum of column_10[30:40] = 1.8589896528353478
@@@ starting subtask column_10[40:50] ...
@@@ returning sum of column_10[40:50] = 1.5358149220974708
@@@ starting subtask column_7[30:40] ...
@@@ returning sum of column_7[30:40] = 1.2842396354359014
@@@ starting subtask column_7[40:50] ...
@@@ returning sum of column_7[40:50] = 1.6320858573474082
@@@ starting subtask column_2[10:20] ...
@@@ returning sum of column_2[10:20] = 0.3838245950380611
@@@ starting subtask column_2[20:30] ...
@@@ returning sum of column_2[20:30] = 0.5661441084018638
@@@ starting subtask column_2[30:40] ...
@@@ returning sum of column_2[30:40] = 0.24546281550649413
@@@ starting subtask column_2[40:50] ...
@@@ starting subtask column_10[10:20] ...
@@@ returning sum of column_10[10:20] = 1.3542075717682014
@@@ returning sum of column_2[40:50] = 0.3704072728971757
@@@ starting subtask column_4[30:40] ...
@@@ returning sum of column_4[30:40] = 1.0063292778549662
@@@ starting subtask column_4[40:50] ...
@@@ returning sum of column_4[40:50] = 0.7566273264555825
@@@ starting subtask column_6[30:40] ...
@@@ returning sum of column_6[30:40] = 1.3247964275079684
@@@ starting subtask column_4[10:20] ...
@@@ returning sum of column_4[10:20] = 0.8661213344691193
@@@ starting subtask column_6[40:50] ...
@@@ returning sum of column_6[40:50] = 1.1977646587411923
@@@ starting subtask column_6[10:20] ...
@@@ returning sum of column_6[10:20] = 1.2186052632178073
@@@ starting subtask column_6[20:30] ...
@@@ returning sum of column_6[20:30] = 1.163378020744928
@@@ starting subtask column_1[30:40] ...
@@@ returning sum of column_1[30:40] = 0.25428835581969733
@@@ starting subtask column_3[10:20] ...
@@@ returning sum of column_3[10:20] = 0.6615422968374268
@@@ starting subtask column_1[40:50] ...
@@@ returning sum of column_1[40:50] = 0.20057735924484543
@@@ starting subtask column_1[10:20] ...
@@@ returning sum of column_1[10:20] = 0.162340920379333
@@@ starting subtask column_1[20:30] ...
@@@ returning sum of column_1[20:30] = 0.15056447817183558
@@@ starting subtask column_4[20:30] ...
@@@ returning sum of column_4[20:30] = 0.568843665271979
@@  Returning sum of 'column_8' = (   8.000)
@@@ starting subtask column_9[10:20] ...
@@  Returning sum of 'column_2' = (   2.000)
@@@ starting subtask column_9[40:50] ...
@@@ returning sum of column_9[40:50] = 1.4998486896979673
@@@ returning sum of column_9[10:20] = 2.2899099993806185
@@@ starting subtask column_9[30:40] ...
@@@ returning sum of column_9[30:40] = 2.041346528847413
@@@ starting subtask column_5[20:30] ...
@@@ returning sum of column_5[20:30] = 1.1346318821802297
@@@ starting subtask column_5[10:20] ...
@@@ returning sum of column_5[10:20] = 1.0089737420839957
@@@ starting subtask column_9[20:30] ...
@@@ returning sum of column_9[20:30] = 1.3708206544463566
@@@ starting subtask column_5[40:50] ...
@@@ returning sum of column_5[40:50] = 0.8420199442536053
@@@ starting subtask column_5[30:40] ...
@@@ returning sum of column_5[30:40] = 0.9480425110538333
@@  Returning sum of 'column_7' = (   7.000)
@@  Returning sum of 'column_10' = (  10.000)
@@  Returning sum of 'column_3' = (   3.000)
@@  Returning sum of 'column_6' = (   6.000)
@@  Returning sum of 'column_4' = (   4.000)
@@  Returning sum of 'column_1' = (   1.000)
@@  Returning sum of 'column_9' = (   9.000)
@@  Returning sum of 'column_5' = (   5.000)
@   Partial sum column_1   =    1.000
@   Partial sum column_2   =    2.000
@   Partial sum column_3   =    3.000
@   Partial sum column_4   =    4.000
@   Partial sum column_5   =    5.000
@   Partial sum column_6   =    6.000
@   Partial sum column_7   =    7.000
@   Partial sum column_8   =    8.000
@   Partial sum column_9   =    9.000
@   Partial sum column_10  =   10.000
@     Total sum            =   55.000
```