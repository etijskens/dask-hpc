# This module decorates the builtin print function to first print an empty line, a line
# `MPI rank: <mpi rank of the current process>`
# and then executes the print statement. In this way it is clear which rank prints what.

from mpi4py import MPI
import io
from pathlib import Path
from datetime import datetime

# remember the undecorated print function
builtin_print = print


def mpi_rank() -> int:
    """Convenience funtion that retrieves the rank of the current process."""
    return MPI.COMM_WORLD.Get_rank()


def mpi_decorator(func):
    """A decorator for the builtin print function that first prints a message identifying the printing rank."""
    # Remark: The wrapper executes two separate calls to builtin_print.
    # That leaves the possibility that the two lines get separated in the output,
    # thus obscuring what is printed by which rank. 
    # We first print to a string and than print the string to stdout.
    def wrapper(*args, **kwargs):
        # remember the 'file' keyword argument if present, to make printing to file and StringIO, ... work too.
        if 'file' in kwargs:
            file = kwargs['file']
            del kwargs['file']
        else:
            file = None
            
        output = io.StringIO()
        builtin_print( f"\nMPI rank: {mpi_rank()} [timestamp: {datetime.now()}]\n"
                      , *args, file=output, **kwargs)
        # probhibit that the line following 'MPI rank: <rank>' starts with a space.
        output = output.getvalue().replace("\n ", "\n")
        builtin_print(output, file=file)

    return wrapper


# Decorate the builtin print function and use it as the default print.
# The undecorated function is still accessible as builtin_print from this module.
# The decorated print funtion accepts all the same arguments. 
print = mpi_decorator(builtin_print)


#==============================================================================
# Q&D test code
if __name__ == "__main__":
    # test print
    # import random, time
    # time.sleep(random.random())
    print(f"-*# rank {mpi_rank()} print to stdout #*-")
    
    # test printing to a file
    with open("mpi_print.output.txt", mode="a") as f:
        print(f"-*# rank {mpi_rank()} print to 'mpi_print.output.txt' #*-", file=f)

