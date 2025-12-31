# Low-Rank Approximations of Matrices
Large matrices often contain redundant information. This means that it is 
possible to form representations of large matrices with less data than what the 
original matrix contains. 
One way of representing a matrix with less data is through
low-rank approximations. 
Generally, low-rank approximations of
a matrix ``A \in \mathbb{R}^{m \times n}`` take two forms: 
``
    A \approx MN,
`` 
where ``M \in \mathbb{R}^{m \times r}`` and ``N \in \mathbb{R}^{r \times n}``,
or 
``
A \approx MBN,
``
where ``M \in \mathbb{R}^{m \times r}``, ``N \in \mathbb{R}^{s \times n}``, and 
``B \in \mathbb{R}^{r \times s}``. 

Once one of the above representations has been obtained they can then be used 
to speed up a number of computations including
matrix multiplication, clustering, or approximate eigenvalue decompositions 
[halko2011finding, eckart1936approximation, udell2019why, park2025curing](@cite).
 
Low rank approximations can also be classified as being either an orthogonal 
form where coordinates are projected perpendicularly onto a hyperplane, or 
an oblique form where points are projected along another plane. 
This distinction can be seen in the figure below where in the left figure the point is 
mapped to the plane ``\mathcal{M}`` so that the path taken is orthogonal to 
``\mathcal{M}``: the plane ``\mathcal{N}`` has no image on the mapping. Alternatively, 
in the oblique projection mapping on the right the point is mapped onto ``\mathcal{M}`` 
along a path parallel to ``\mathcal{Z}``.  
```@raw html
<img src="../images/projection.svg" width =400 height = 300/> 
```

We also can consider low-rank approximations for symmetric matrices and general matrices.
For symmetric and general matrices, the RandomizedSVD can be used as the orthogonal 
projection method [halko2011finding](@cite).   

As far as oblique methods go, the difference between symmetric and asymmetric decompositions
becomes more complicated. 
For symmetric matrices, the go-to approximation is the Nystrom
approximation. 
For non-symmetric matrices, we can have a generalization of Nystrom 
known as Generalized Nystrom or we can use interpolative decompositions (IDs), 
which select subsets of 
the rows and/or columns of a matrix. 
If these interpolative decompositions are performed 
to select only columns or only rows then they are known as one-sided IDs. 
Otherwise, if they select both columns and rows then they are known as a CUR decomposition. Below, we 
present a summary of the decompositions in a table. 

|Approximation Name| General Matrices| Interpolative| Type| Form of Approximation|
|:-----------------|:----------------|:-------------|:----|:---------------------|
|RandRangeFinder| Yes| No| Orthogonal| ``A \approx QQ^\top A``|
|RandSVD|Yes|No|Orthogonal|``A \approx U \Sigma V^\top``|
|Nystrom| Symmetric| Can be| Oblique| ``(AS)((AS)^\top AS)^\dagger(AS)^\top``|
|Generalized Nystrom| Yes| Can be| Oblique| ``(AS_1)(S_2A AS_1)^\dagger S_2 A``|
|CUR| Yes| Yes| Oblique| ``(A[:,J])U(A[I,:])``|
|One-Sided ID| Yes| Yes| Oblique| ``A[:,J]U_c`` or ``U_r A[I,:]``|

In `RLinearAlgebra`, 
once you have obtained a low-rank approximation `Recipe`, you can use it 
to do multiplications (via `mul!`) and/or left inversion (via `ldiv!`). 
Below we have the table of approximation recipes 
and indicate how they can be used.

|Approximation Name| `mul!`| `ldiv!`|
|:-----------------|:------|:-------|
|RandRangeFinderRecipe| Yes| No|
|RandSVDRecipe|Yes| No|
|NystromRecipe|Yes| No|
|CURRecipe|Yes| No|
|IDRecipe (One-Sided ID)|Yes|No|

# The Randomized Rangefinder
The idea behind the randomized range finder is to find 
an orthogonal matrix ``Q`` such that
``A \approx QQ^\top A``. [halko2011finding](@citet) showed that 
forming ``Q`` was as simple as compressing ``A`` from the right  
and storing the ``Q`` from the
resulting QR factorization. 
Despite the simplicity of this procedure,
 if the compression dimension, ``k``, exceeds 2, then
``\|A - QQ^\top A\|_F \leq \sqrt{k+1} (\sum_{i=k+1}^{\min{(m,n)}}\sigma_{i})^{1/2}``, 
where ``\sigma_{k+1}`` is the ``k+1^\text{th}`` singular value of ``A`` 
[halko2011finding; Theorem 10.5](@cite). 
This is very close to the error from the truncated SVD, which 
is known to be the lowest achievable error. 
For matrices whose singular values decay quickly, this bound can be 
conservative in comparison to observed performance. 
However, for some matrices whose singular values 
decay slowly this bound is fairly tight. 
`RLinearAlgebra` implements the randomized rangefinder using 
[`RangeFinder`](@ref).

Power iterations can be added to improve the quality of the approximation.
Power iterations basically involve multiplying the matrix with itself, 
which results in raising each singular value to a 
higher power. This powering of the singular values increases the gap between the 
singular values making them easier to resolve.
In `RLinearAlgebra`, you can control the number of power iterations 
using the `power_its` keyword in [`RangeFinder`](@ref).

One issue with power iterations is that they can sometimes be 
unstable. 
We can improve the stability of these iterations by orthogonalizing between
power iterations (akin to Arnoldi iterations).
In `RLinearAlgebra` you can control whether or not the
orthogonalization is performed using the `orthogonalize` keyword argument in the 
[`RangeFinder`](@ref). 

!!! info 
    If the cardinality of the compressor in the `RangeFinder` is not `Right()` a warning 
    will be returned and the approximation may be inefficient or less accurate.  

## A RangeFinder Example
Lets say that we wish to obtain a five-dimensional Randomized Rangefinder approximation 
to matrix with 1000 rows and columns and a rank of five.
In `RLinearAlgebra`, we can easily do this with the Fast Johnson-Lindenstrauss
Compressor (see [`FJLT`](@ref)), say, as follows.

```julia
using RLinearAlgebra, LinearAlgebra

# Generate the matrix we wish to approximate
A = randn(1000, 5) * randn(5, 1000);

# Form the RangeFinder Structure
approx = RangeFinder(
    compressor = FJLT(compression_dim = 5, cardinality = Right())
)

# Approximate A
range_A = rapproximate(approx, A)

# Check the error of the approximation
norm(A - range_A * (range_A' * A))
```

### Adding Power Iterations with and without Orthogonalization

To see the benefits of power iterations, consider the same example but now with 
`compression_dim = 3`. 

Below, we first compute the best possible truncation error using 
a singular value decomposition of rank 3.
Then, we compute the Randomized Rangefinder _without_ the power iteration.
Then, we compute the Randomized Rangefinder _with_ the power iteration.
Finally, we compute the Randomized Rangefinder _with_ the power iteration _and_ 
orthogonalization.

```julia
# Get error of truncated svd by computing the sqrt of the sum^2 of singular values 4:1000
printstyled("Error of rank 3 truncated SVD:",
    sqrt(sum(svd(A).S[4:end].^2)),
    "\n"
)

# Try approximating with a compression dimension of 3 and no power its/orthogonalization 
# Form the RangeFinder Structure
approx = RangeFinder(
    compressor = FJLT(compression_dim = 3, cardinality = Right())
);

range_A = rapproximate(approx, A);

printstyled("Error of rank 3 approximation:",
    norm(A - range_A * (range_A' * A)),
    "\n"
)

# Now consider adding power iterations 
approx_pi = RangeFinder(
    compressor = FJLT(compression_dim = 3, cardinality = Right()),
    power_its = 10
);

range_A_pi = rapproximate(approx_pi, A);


printstyled("Error with 10 Power its and Orthogonalization:",
    norm(A - range_A_pi * (range_A_pi' * A)),
    "\n"
)

# Now consider power its with orthogonalization
approx_pi_o = RangeFinder(
    compressor = FJLT(compression_dim = 3, cardinality = Right()),
    power_its = 10,
    orthogonalize = true
);

range_A_pi_o = rapproximate(approx_pi_o, A);

printstyled("Error with 10 Power its and Orthogonalization:",
    norm(A - range_A_pi_o * (range_A_pi_o' * A)),
    "\n"
)
```
# The RandSVD
The RandomizedSVD is a form of low-rank approximation that returns the approximate 
singular values and vectors of the truncated SVD. Algorithmically, it is implemented as
three additional steps to the Randomized Rangefinder in [halko2011finding](@cite). 
Specifically these steps are:
1. Take the ``Q`` matrix from the Randomized Rangefinder and compute ``Q^\top A``.  
2. Compute the ``W,S,V = \text{svd}(Q^\top A)``.
3. Obtain the left singular vectors from ``U = Q^\top W``.

Since, the RandomizedSVD is built on the Randomized RangeFinder, the effects
of all modifications, such as power iterations and orthogonalization still apply.



## A RandSVD example
We demonstrate how to generate a rank 5 RandomziedSVD in `RLinearAlgebra` with the 
Fast Johnson-Lindenstrauss compressor (see [`FJLT`](@ref)).
See [`RandSVD`](@ref) for interface details.

```julia
using RLinearAlgebra, LinearAlgebra

# Generate the matrix we wish to approximate
A = randn(1000, 5) * randn(5, 1000);

# Form the RangeFinder Structure
approx = RandSVD(
    compressor = FJLT(compression_dim = 5, cardinality = Right())
)

# Approximate A
randsvd_A = rapproximate(approx, A)

# Compare singular vectors
svd(A).S[1:5]

randsvd_A.S

# Compare multiplications
x = rand(1000);

norm(A * x - randsvd_A * x)
```

!!! info
    As for the RandomizedSVD if the cardinality of the compressor is not `Right()` a warning will be returned and the approximation may be incorrect.
