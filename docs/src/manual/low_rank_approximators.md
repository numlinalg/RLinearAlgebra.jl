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
becomes more complicated. For symmetric matrices, the go to approximation is the Nystr\"om
approximation. For the non-symmetric matrices, we can have a generalization of Nystr\"om 
known as Generalized Nystr\"om or we can interpolative approaches, which select subsets of 
the rows and/or columns to a matrix. If it these interpolative decompositions are performed 
to select only columns or only rows then they are known as one sided IDs, if they are used 
to select both columns and rows then they are known as a CUR decomposition. Below, we 
present a summary of the decompositions in a table. 
|Approximation Name| General Matrices| Interpolative| Type| Form of Approximation|
|:-----------------|:----------------|:-------------|:----|:---------------------|
|RandRangeFinder| Yes| No| Orthogonal| ``A \\approx QQ^\top A``|
|RandSVD|Yes|No|Orthogonal|``A \\approx U\\SigmaV^\\top``|
|Nystr\"om| Symmetric| Can be| Oblique| ``(AS)((SA)^\\top AS)^\\dagger(AS)^\\top``|
|Generalizedd Nystr\"om| Yes| Can be| Oblique| ``(AS_1)(S_2A AS_1)^\\dagger S_2 A``|
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
|Nystr\"om|Yes| No|
|CUR|Yes| No|
|One-Sided-ID|Yes|No|
# A RangeFinder Example
Lets say that we wish to obtain a rank-5 RandomizedSVD to matrix with 1000 rows and columns.
In RLinearAlgebra.jl we can do this by first generating the `RandomizedSVD` `Approximator`.
This will require us to specify a `Compressor` with the desired rank of approximation as the
`compression_dim` and the `cardinality=Right()`, the number of power iterations we want 
to be performed, and the type of power iterations we want to perform. 
