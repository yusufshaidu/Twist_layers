# Twist_layers

This code generate configurations for twisted bilayer TMDs and graphene.
We follow closely the approach presented in PHYSICAL REVIEW B 90, 155451 (2014)
use case: twisted bilayer graphene, bilayer TMD, TMD on hBN substrate ...

# Heterostructure

For heterostructures, we write different set of n and m for the two layers and search for the smallest positive integers
that gives the specified angle and has strain less than the user specified threshold.

# For a general case

We implement a code for a generate twisted case following the report in https://arxiv.org/pdf/2104.09591

We have implemented two parallelization scheme namely: (1) using numba and (2) writing a c++ inerface and employing openmpi pragma

# Running the code

The code reads a yaml file that contains information about the two layers. Please check the example.yaml for reference.

To run the code use the following command:

```python3 path/to/twistlayers.py -c file.yaml```
