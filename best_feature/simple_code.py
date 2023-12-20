from mpi4py import MPI

assert MPI.COMM_WORLD.Get_size() > 1

rank = MPI.COMM_WORLD.Get_rank()

if rank == 0:
    MPI.COMM_WORLD.send(None, dest=1, tag=42)

elif rank == 1:
    MPI.COMM_WORLD.recv(source=0, tag=42)
