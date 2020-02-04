#! /bin/bash
#COBALT -A CSC249ADCD08 --attrs enable_ssh=1

# Load up the Python environment
module load miniconda-3/latest
source activate /lus/theta-fs0/projects/CSC249ADCD08/design/graph_sage/env
export PYTHONPATH=""  ## Get rid of the  default path from the modules

# Set config and run
#  - Turning off XLA to see if that gives more threads. No effect
#  - Adding prefetching
export KMP_BLOCKTIME=0
export KMP_AFFINITY="granularity=fine,compact,1,0"
export MPICH_GNI_FORK_MODE=FULLCOPY
export OMP_NUM_THREADS=64

aprun -n 1 -N 1 -d $OMP_NUM_THREADS -j 1 --cc depth python profile_train.py $@
