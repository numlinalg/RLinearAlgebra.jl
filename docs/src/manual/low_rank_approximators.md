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

As far as oblique methods go, the difference between symmetric and asymmetric decompositions
becomes more complicated. For symmetric matrices, the go to approximation is the Nystrom
approximation. For the non-symmetric matrices, we can have a generalization of Nystrom 
known as Generalized Nystrom or we can interpolative approaches, which select subsets of 
the rows and/or columns to a matrix. If it these interpolative decompositions are performed 
to select only columns or only rows then they are known as one sided IDs, if they are used 
to select both columns and rows then they are known as a CUR decomposition. Below, we 
present a summary of the decompositions in a table. 

|Approximation Name| General Matrices| Interpolative| Type| Form of Approximation|
|:-----------------|:----------------|:-------------|:----|:---------------------|
|RandRangeFinder| Yes| No| Orthogonal| ``A \approx QQ^\top A``|
|RandSVD|Yes|No|Orthogonal|``A \approx U \Sigma V^\top``|
|Nystrom| Symmetric| Can be| Oblique| ``(AS)((SA)^\top AS)^\dagger(AS)^\top``|
|Generalizedd Nystrom| Yes| Can be| Oblique| ``(AS_1)(S_2A AS_1)^\dagger S_2 A``|
|CUR| Yes| Yes| Oblique| ``(A[:,J])U(A[I,:])``|
|One-Sided-ID| Yes| Yes| Oblique| ``A[:,J]U_c`` or ``U_r A[I,:]``|

Once you have obtained a low-rank approximation you can then use it to perform 
multiplications in all cases or in some specific areas use it to precondition a linear 
system through the ldiv! function. Below we have the table of decompositions and indicate
how they can be used.

|Approximation Name| `mul!`| `ldiv!`|
|:-----------------|:------|:-------|
|RandRangeFinder| Yes| No|
|RandSVD|Yes| No|
|Nystrom|Yes| No|
|CUR|Yes| No|
|One-Sided-ID|Yes|No|
# The Randomized Rangefinder
The idea behind the randomized range finder is to find an orthogonal matrix ``Q`` such that
``A \approx QQ^\top A``. In their seminal work, [halko2011finding](@cite) showed that 
forming ``Q`` was as simple as compressing ``A`` from the right  and storing the Q from the
resulting QR factorization. Despite the simplicity of this procedure they were able to show
 if the compression dimension, ``k>2``, then
``\|A - QQ^\top A\|_F \leq \sqrt{k+1} (\sum_{i=k+1}^{\min{(m,n)}}\sigma_{i})^{1/2}``, 
    where ``\sigma_{k+1}`` is the ``k+1^\text{th}`` singular value of A (see Theorem 10.5 
of [halko2011finding](@cite)). This is very close to the error from the truncated SVD, which 
is known to be the lowest achievable error. 


For many matrices that singular values that decay quickly, this bound can be far more 
conservative than the observed performance. However, for some matrices whose singular values 
decay slowly this bound is fairly tight. Luckily, using power iterations we can 
still improve the quality of the approximation. Power Iterations basically involve
multiplying the matrix with itself, which results in raising each singular value to a 
higher power. This powering of the singular values increases the gap between the singular 
values large making them easier to differentiate between each other and therefore capture.
In `RLinearAlgebra`, you can control the number of power iterations using the `power_its`
keyword in the constructor. 

One issue with power iterations is that they can sometimes be 
unstable. We can also improve the stability of these iterations by orthogonalizing between
power iterations. Meaning that instead of computing ``A A^\top A`` as is done in the power 
iterations we compute ``A^\top A`` and take a QR factorization of this matrix to obtain a 
``Q`` then compute ``A Q``. In RLinearAlgebra you can control whether or not the
orthogonalization is performed using the `orthogonalize` keyword argument in the 
constructor. 

## A RangeFinder Example
Lets say that we wish to obtain a rank-5 RandomizedSVD to matrix with 1000 rows and columns.
In RLinearAlgebra.jl we can do this by first generating the `RandomizedSVD` `Approximator`.
This will require us to specify a `Compressor` with the desired rank of approximation as the
`compression_dim` and the `cardinality=Right()`, the number of power iterations we want 
to be performed, and the type of power iterations we want to perform. 

```julia
# Generate the matrix we wish to approximate
A = randn(1000, 5) * randn(5, 1000);


```

# The RandSVD

## A RandSVD example
