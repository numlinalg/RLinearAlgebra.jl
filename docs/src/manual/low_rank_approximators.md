# Low-Rank Approximations of Matrices
Often large matrices contain a lot of redundant information. This means that it is often 
possible to form representations of large matrices with far fewer vectors than what the 
original matrix contains. Representing a matrix with far fewer vectors than what it 
initially contains is known as low-rank approximation. Generally, low-rank approximations of
a matrix ``A \in \mathbb{R}^{m \times n}``take two forms either a two matrix form where
``
    A \approx MN,
`` 
where ``M \in \mathbb{R}^{m \times r}`` and ``N \in \mathbb{R}^{r \times n}``,
or the three matrix representation where 
``
A \approx MBN
``
and ``M \in \mathbb{R}^{m \times r}``, ``N \in \mathbb{R}^{t \times n}``, and 
``B \in \mathbb{R}^{r \times s}``. 

Once one of the above representations has been obtained they can then be used to speed up:
matrix multiplication, clustering, or approximate eigenvalue decompositions [Add citations].
 
Low rank approximations can take two different forms one being the orthogonal projection 
form where coordinates are projected perpendicularly to onto a plane and the second being
the oblique forms where points are projected along another plane (see the below figure 
for a visualization).
```@raw html
<img src="../images/projection.png" width =400 height = 300/> 
```

We also can consider low-rank approximations for symmetric matrices and general matrices.
For symmetric and general matrices, the RandomizedSVD can be used as the orthogonal 
projection method [halko2011finding](@cite). 
# A RangeFinder Example
Lets say that we wish to obtain a rank-5 RandomizedSVD to matrix with 1000 rows and columns.
In RLinearAlgebra.jl we can do this by first generating the `RandomizedSVD` `Approximator`.
This will require us to specify a `Compressor` with the desired rank of approximation as the
`compression_dim` and the `cardinality=Right()`, the number of power iterations we want 
to be performed, and the type of power iterations we want to perform. 
