#! /bin/bash
#COBALT -A CSC249ADCD08 --attrs enable_ssh=1

# Load up the Python environment
module load miniconda-3/latest
source activate /lus/theta-fs0/projects/CSC249ADCD08/design/graph_sage/env
export PYTHONPATH=""  ## Get rid of the  default path from the modules

# Read the number of nodes and ranks per node
nodes=$1
ranks_per_node=$2
threads_per_core=$3
total_ranks=$((nodes * ranks_per_node))
threads_per_rank=$(((64 * threads_per_core) / ranks_per_node))

# Set config and run
#  - Turning off XLA to see if that gives more threads. No effect
#  - Adding prefetching
export KMP_BLOCKTIME=0
export KMP_AFFINITY="granularity=fine,compact,1,0"
export MPICH_GNI_FORK_MODE=FULLCOPY
export OMP_NUM_THREADS=$threads_per_rank

aprun -n $total_ranks -N $ranks_per_node -d $OMP_NUM_THREADS -j $threads_per_core --cc depth python profile_train.py ${@:4}
