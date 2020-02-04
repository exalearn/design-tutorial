# design-tutorial

Tutorial materials used by Design subgroup for the ECP Annual Meeting

## Installation

This project requires Anaconda and git to perform most of the demo steps.

1. Clone this repository from GitHub. Note that you will need Git LFS to access the training data for the tutorial.
1. Move to the root directory for the tutorial: `cd design-tutorial`
1. Install the environment with Anaconda: `conda env create --file environment.yml --force`. 
    a. If you would like GPU support, change the `tensorflow` dependency to `tensorflow-gpu`
    b. If you are running at an HPC center, consult your facility documentation for how to configure Anaconda environments.
    
Your environment will be complete with the libraries needed to run everything except the distributed training with Horovod.
We recommend consulting your facility's documentation about Horvod. 

### ALCF Installation

At ALCF, you can access Horovod and Tensorflow using the environments already available with the Anaconda module at ALCF.

1. Activate the ALCF Anaconda module: `module load miniconda-3/latest`
2. Clone the base environment, which contains ALCF provided versions of Horovod and Tensorflow: `conda create --clone base -p ./env`
3. Follow Anaconda's directions for activating the environment: `source activate ./env`
4. Install the packages specific to our environment: `conda env update --file envs/alcf.yml`

### Running on Bigger Dataset

If you would like to run the tutorial on the full dataset, you can find it on the [Database of Water Cluster Minima](https://sites.uw.edu/wdbase/database-of-water-clusters/).
