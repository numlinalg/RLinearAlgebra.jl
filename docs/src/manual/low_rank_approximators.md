# Low-Rank Approximations of Matrices
Often large matrices contain a lot of redundant information. 

# A RangeFinder Example
Lets say that we wish to obtain a rank-5 RandomizedSVD to matrix with 1000 rows and columns.
In RLinearAlgebra.jl we can do this by first generating the `RandomizedSVD` `Approximator`.
This will require us to specify a `Compressor` with the desired rank of approximation as the
`compression_dim` and the `cardinality=Right()`, the number of power iterations we want 
to be performed, and the type of power iterations we want to perform. 
