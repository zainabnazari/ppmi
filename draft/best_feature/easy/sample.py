import os
from dask_mpi import initialize
from dask.distributed import Client
import joblib
import sklearn.datasets
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":

    print('I am before client initialization')

    # Initialize Dask cluster and client interface
    n_tasks = int(os.getenv('SLURM_NTASKS'))
    mem = os.getenv('SLURM_MEM_PER_CPU')
    mem = str(int(mem)) + 'MB'

    initialize(memory_limit=mem)

    dask_client = Client()

    dask_client.wait_for_workers(n_workers=(n_tasks - 2))
    # dask_client.restart()

    num_workers = len(dask_client.scheduler_info()['workers'])
    print("%d workers available and ready" % num_workers)

    # Generate sample data
    X, y = sklearn.datasets.make_classification()

    # Use joblib with Dask backend
    with joblib.parallel_backend("dask"):
        RandomForestClassifier().fit(X, y)

