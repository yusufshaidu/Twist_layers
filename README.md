# Twist_layers

This code generate configurations for twisted bilayer TMDs and graphene.
We follow closely the approach presented in PHYSICAL REVIEW B 90, 155451 (2014)
use case: twisted bilayer graphene, bilayer TMD, TMD on hBN substrate ...

# Heterostructure

For heterostructures, we write different set of n and m for the two layers and search for the smallest positive integers
that gives the specified angle and has strain less than the user specified threshold.

# For a general case

I implement a code for a generate twisted case following the report in https://arxiv.org/pdf/2104.09591
-- Note: This code is slower because of the $N^8$ scaling to generate the 8 indices for the moire supercell.
We have implemented two parallelization scheme namely: (1) using numba and (2) writing a c++ inerface and employing openmpi pragma

On my computer, with 4 processors, numba is twice faster than the cpp code.

# Running the code

The codes reads a yaml file that contains information about the two layers. Please check the example.yaml for reeference.

To run the code use the following command:

```python3 path/to/generate.py -c file.yaml
