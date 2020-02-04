"""Evaluate model training performance"""

from graphsage.mpnn.data import make_data_loader
from graphsage.mpnn.layers import custom_objects
from graphsage.utils import get_platform_info

import horovod.tensorflow.keras as hvd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

from argparse import ArgumentParser
from datetime import datetime
from time import perf_counter
from tqdm import tqdm
import json
import os


# Get the rank information
hvd.init()
rank = hvd.rank()
size = hvd.size()


# Hard-coded paths for data and model
_data_path = os.path.join('..', '..', 'data', 'output', 'water_clusters.proto')
_model_path = os.path.join('..', 'model.h5')


if __name__ == "__main__":
    # Parse the arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--batch-sizes', '-b', nargs='*', default=[32], help='Batch size for each rank', type=int)
    arg_parser.add_argument('--n_batches', default=512, help='Number of batches to use in testing', type=int)
    arg_parser.add_argument('--parallel-loader', '-p', help='Number of threads to use in data loader steps',
                            type=int, default=tf.data.experimental.AUTOTUNE)
    arg_parser.add_argument('--inter_op', help='Number of inter_op threads to use', type=int, default=0)

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__

    # Get the host and comm pool information
    host_info = get_platform_info()
    host_info['total_ranks'] = size
    host_info['ranks_per_node'] = hvd.local_size()

    # Configure Tensorflow
    #  Allow for GPU memory growth
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    #  Set the CPU parallelism
    n_threads = host_info['accessible_cores']  # Make sure
    tf.config.threading.set_intra_op_parallelism_threads(n_threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.inter_op)

    # Open an experiment directory
    start_time = datetime.utcnow()
    out_dir = f'{start_time.strftime("%d%b%y-%H%M%S")}-{host_info["hostname"]}'
    if rank == 0:
        os.makedirs(out_dir)

    # Save the parameters and host information
    if rank == 0:
        with open(os.path.join(out_dir, 'host_info.json'), 'w') as fp:
            json.dump(host_info, fp, indent=2)
        with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
            json.dump(run_params, fp, indent=2)

    # Load the model
    model = load_model(_model_path, custom_objects=custom_objects)
    opt = hvd.DistributedOptimizer(Adam())
    model.compile(opt, 'mean_squared_error')
    
    # Make the needed callbacks to sharing examples
    callbacks = [
        hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]

    # Loop over batch sizes
    rates = []
    for b in tqdm(args.batch_sizes):
        # Make the data loader
        loader = make_data_loader(_data_path, batch_size=b, n_threads=args.parallel_loader).prefetch(16)

        # Run two batches for burn in, which forces TF to compile the model
        model.fit(loader, epochs=1, shuffle=False, steps_per_epoch=2, verbose=0, callbacks=callbacks)

        # Run a bunch of batches and time them
        start_time = perf_counter()
        model.fit(loader, epochs=1, shuffle=False, steps_per_epoch=args.n_batches, verbose=0, callbacks=callbacks)
        rates.append(args.n_batches / (perf_counter() - start_time))

    if rank == 0:
        with open(os.path.join(out_dir, 'timings.json'), 'w') as fp:
            json.dump({
                'batch_size': args.batch_sizes,
                'rate': rates
            }, fp, indent=2)
