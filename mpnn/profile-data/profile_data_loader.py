"""Trains the model using Keras without any kind of data-parallel training"""

from graphsage.mpnn.data import parse_records, prepare_for_batching, combine_graphs, make_training_tuple
from graphsage.utils import get_platform_info

import tensorflow as tf
tf.enable_eager_execution()

from argparse import ArgumentParser
from time import perf_counter
from shutil import copyfile
from tqdm import tqdm
import json
import os


# Control growth on GPU
tf_config=tf.ConfigProto()
tf_config.gpu_options.allow_growth=True
sess = tf.Session(config=tf_config)

# Hard-coded paths for data and model
_val_path = 'water_clusters.proto.gz'
_uncompressed_path = os.path.join('..', '..', 'data', 'output', 'water_clusters.proto')
_num_test_batches = 256


if __name__ == "__main__":
    # Parse the arguments
    arg_parser = ArgumentParser()
    arg_parser.add_argument('--batch-sizes', '-b', nargs='*', default=[32], help='Batch size', type=int)
    arg_parser.add_argument('--test-batches', '-i', default=256, help='Number of test batches', type=int)
    arg_parser.add_argument('--parallel', '-p', default=4, help='Number of IO parallel threads', type=int)
    arg_parser.add_argument('--shm', action='store_true', help='Test moving data to SHM')

    # Parse the arguments
    args = arg_parser.parse_args()
    run_params = args.__dict__

    # Make the testing function
    def test_data_loader(loader: tf.data.TFRecordDataset, n_test: int = args.test_batches) -> float:
        """Evaluate how fast it is to read the entire dataset

        Args:
            loader: Data loader to evaluate
            n_test (int): Number of batches to use for the test
        Returns:
            (float): Batch processing rate (batches / s)
        """
        start_time = perf_counter()
        for i, _ in zip(range(n_test), loader):
            continue
        return n_test / (perf_counter() - start_time)

    # Get the host information
    host_info = get_platform_info()

    # Open an experiment directory
    out_dir = f'{host_info["hostname"]}'
    os.makedirs(out_dir, exist_ok=True)

    # Save the parameters and host information
    with open(os.path.join(out_dir, 'host_info.json'), 'w') as fp:
        json.dump(host_info, fp, indent=2)
    with open(os.path.join(out_dir, 'run_params.json'), 'w') as fp:
        json.dump(run_params, fp, indent=2)

    # Standard data transformation
    rates = []
    for b in tqdm(args.batch_sizes, desc='Standard'):
        r = tf.data.TFRecordDataset(_val_path, 'GZIP').batch(b).map(parse_records).map(prepare_for_batching). \
            map(combine_graphs).map(make_training_tuple)
        rates.append(test_data_loader(r))
    with open(os.path.join(out_dir, 'standard.json'), 'w') as fp:
        json.dump({
            'description': 'No parallelism, GZIPed data, data on disk',
            'batch_sizes': args.batch_sizes,
            'rates': rates
        }, fp, indent=2)

    # Reading from uncompressed file
    rates = []
    for b in tqdm(args.batch_sizes, desc='No compression'):
        r = tf.data.TFRecordDataset(_uncompressed_path).batch(b).map(parse_records).map(prepare_for_batching). \
            map(combine_graphs).map(make_training_tuple)
        rates.append(test_data_loader(r))
    with open(os.path.join(out_dir, 'no-compression.json'), 'w') as fp:
        json.dump({
            'description': 'No parallelism, data on disk',
            'batch_sizes': args.batch_sizes,
            'rates': rates
        }, fp, indent=2)

    # Prefetching
    rates = []
    for b in tqdm(args.batch_sizes, desc='Prefectching'):
        r = tf.data.TFRecordDataset(_uncompressed_path).batch(b).map(parse_records).map(prepare_for_batching). \
            map(combine_graphs).map(make_training_tuple).prefetch(8)
        rates.append(test_data_loader(r))
    with open(os.path.join(out_dir, 'prefetching.json'), 'w') as fp:
        json.dump({
            'description': 'No parallelism, prefetching, data on disk',
            'batch_sizes': args.batch_sizes,
            'rates': rates
        }, fp, indent=2)

    # Parallelism
    rates = []
    para = tf.data.experimental.AUTOTUNE
    for b in tqdm(args.batch_sizes, desc='Parallel Autotune'):
        r = tf.data.TFRecordDataset(_uncompressed_path).batch(b).map(parse_records, para)\
            .map(prepare_for_batching, para). \
            map(combine_graphs, para).map(make_training_tuple, para).prefetch(8)
        rates.append(test_data_loader(r))
    with open(os.path.join(out_dir, 'parallel-autotune.json'), 'w') as fp:
        json.dump({
            'description': 'Autotune parallel, prefetching, data on disk',
            'batch_sizes': args.batch_sizes,
            'rates': rates
        }, fp, indent=2)

    rates = []
    para = args.parallel
    for b in tqdm(args.batch_sizes, desc=f'Parallel Fixed {para}'):
        r = tf.data.TFRecordDataset(_uncompressed_path).batch(b).map(parse_records, para)\
            .map(prepare_for_batching, para). \
            map(combine_graphs, para).map(make_training_tuple, para).prefetch(8)
        rates.append(test_data_loader(r))
    with open(os.path.join(out_dir, f'parallel-fixed-{para}.json'), 'w') as fp:
        json.dump({
            'description': f'Parallel with threads fixed at {para}, prefetching, data on disk',
            'batch_sizes': args.batch_sizes,
            'rates': rates
        }, fp, indent=2)

    # Test in SHM
    if not args.shm:
        exit()

    data_path = os.path.join('/dev', 'shm', 'data')
    try:
        copyfile(_uncompressed_path, data_path)
        rates = []
        for b in tqdm(args.batch_sizes, desc='Data in SHM'):
            r = tf.data.TFRecordDataset(data_path).batch(b).map(parse_records, para)\
                .map(prepare_for_batching, para). \
                map(combine_graphs, para).map(make_training_tuple, para).prefetch(8)
            rates.append(test_data_loader(r))
        with open(os.path.join(out_dir, 'parallel-shm.json'), 'w') as fp:
            json.dump({
                'description': 'Fixed parallel, prefetching, data in memory',
                'batch_sizes': args.batch_sizes,
                'rates': rates
            }, fp, indent=2)
    finally:
        os.unlink(data_path)

