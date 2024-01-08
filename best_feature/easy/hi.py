from mpi4py import MPI
from dask_mpi import initialize
from dask.distributed import Client
import dask.array as da

def main():
    # Initialize Dask for MPI
    initialize()

    # Create a Dask client
    client = Client()

    # Get MPI information
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Create a Dask array
    shape = (1000, 1000)
    chunks = (500, 500)
    x = da.ones(shape, chunks=chunks)

    # Perform a simple computation
    result = (x + x.T).sum()

    # Gather results to the root process
    result_sum = comm.reduce(result, op=MPI.SUM, root=0)

    # Print results
    if rank == 0:
        print("Total sum across all processes:", result_sum)

if __name__ == "__main__":
    main()

