# Low-Rank Approximations of Matrices
Often large matrices contain a lot of redundant information. This means that it is often 
possible to form representations of large matrices with far fewer vectors than what the 
original matrix contains. Representing a matrix with a small number of vectors 
is known as low-rank approximation. Generally, low-rank approximations of
a matrix ``A \in \mathbb{R}^{m \times n}`` take two forms either a two matrix form where
``
    A \approx MN,
`` 
where ``M \in \mathbb{R}^{m \times r}`` and ``N \in \mathbb{R}^{r \times n}``,
or the three matrix form where 
``
A \approx MBN
``
and ``M \in \mathbb{R}^{m \times r}``, ``N \in \mathbb{R}^{s \times n}``, and 
``B \in \mathbb{R}^{r \times s}``. 

Once one of the above representations has been obtained they can then be used to speed up:
matrix multiplication, clustering, or approximate eigenvalue decompositions 
[halko2011finding, eckart1936approximation, udell2019why, park2025curing](@cite).
 
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

In RLinearAlgebra, 
once you have obtained a low-rank approximation `Recipe` you can then use it to perform 
multiplications in all cases or in some specific areas use it to precondition a linear 
system through the `ldiv!` function. Below we have the table of approximation recipes 
and indicate how they can be used.

|Approximation Name| `mul!`| `ldiv!`|
|:-----------------|:------|:-------|
|RandRangeFinderRecipe| Yes| No|
|RandSVDRecipe|Yes| No|
|NystromRecipe|Yes| No|
|CURRecipe|Yes| No|
|IDRecipe(One-Sided-ID)|Yes|No|
# The Randomized Rangefinder
The idea behind the randomized range finder is to find an orthogonal matrix ``Q`` such that
``A \approx QQ^\top A``. In their seminal work [halko2011finding](@cite) showed that 
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
values making them easier to accurately capture.
In `RLinearAlgebra`, you can control the number of power iterations using the `power_its`
keyword in the constructor. 

One issue with power iterations is that they can sometimes be 
unstable. We can also improve the stability of these iterations by orthogonalizing between
power iterations. Meaning that instead of computing ``A A^\top A`` as is done in the power 
iterations we compute ``A^\top A`` and take a QR factorization of this matrix to obtain a 
``Q`` then compute ``A Q``. In RLinearAlgebra you can control whether or not the
orthogonalization is performed using the `orthogonalize` keyword argument in the 
constructor. 

!!! info 
    If the cardinality of the compressor in the `RangeFinder` is not `Right()` a warning 
    will be returned and the approximation may be incorrect.  

## A RangeFinder Example
Lets say that we wish to obtain a rank-5 RandomizedSVD to matrix with 1000 rows and columns.
In RLinearAlgebra.jl we can do this by first generating the `RandomizedSVD` `Approximator`.
This will require us to specify a `Compressor` with the desired rank of approximation as the
`compression_dim` and the `cardinality=Right()`, the number of power iterations we want 
to be performed, and whether we want to orthogonalize the power iterations. 


To begin, we consider forming an approximation to a rank 5 matrix with 1000 rows and 
columns. Then will define a `RangeFinder` structure with a `FJLT` compressor with a  
`compression_dim = 5` and a 
`cardinality = Right()`. After defining this structure we will then use `rapproximate` to 
generate a `RangeFinderRecipe`, which we will then compute the error relying on the ability 
to multiply `RangeFinderRecipe`s.
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
To see the benefits of power iterations we consider the same example but now with 
`compression_dim = 3`, then we consider the truncated SVD error, 
the error of an approximation with no power iterations, 
the error of an approximation with 10 power iterations but no
orthogonalization, and the error of an approximation with 10 power iterations and 
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
singular values and vectors to the truncated SVD. Algorithmically, it is implemented as
three additional steps to the Randomized Rangefinder in [halko2011finding](@cite). 
Specifically these steps are:
1. Take the ``Q`` matrix from the Randomized Rangefinder and compute ``Q^\top A``.  
2. Compute the ``W,S,V = \text{svd}(Q^\top A)``.
3. Obtain the left singular vectors from ``U = Q^\top W``.

Since, the RandomizedSVD is simply an extension of the Randomized RangeFinder, the effects
of all modifications, such as power iterations and orthogonalization still apply.  The 
difference between the two procedures is found in the Recipes. Where for the `RandSVDRecipe`
you find a approximate truncated SVD where the singular values can be accessed by 
calling `recipe.S`, the left singular vectors can be accessed by calling `recipe.U`,
and the right singular vectors can be accessed by calling `recipe.V`. Additionally, 
when you multiply with the RandomizedSVD it is as if you are multiplying with the 
truncated SVD, meaning for a vector ``x`` the operation ``USV^\top x`` is performed. This 
type of multiplication can be substantially faster than multiplications with the original 
matrix.

!!! info
    As for the RandomizedSVD if the cardinality of the compressor is not `Right()` a warning 
    will be returned and the approximation may be incorrect.

## A RandSVD example
We now demonstrate how to use the RandSVD, by first generating the technique structure 
with a `FJLT` compressor with `compression_dim = 5` and `cardinality = Right()`. Then we 
will run `rapproximate` and compare the singular values of the returned recipe to the 
5 singular values of the truncated SVD. We will then end the experiment by comparing 
the difference between multiplying a our `RandSVDRecipe` to vector and multiplying the 
original matrix.

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
