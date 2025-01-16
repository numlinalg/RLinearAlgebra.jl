# Date: 01/14/2025
# Author: Christian Varner
# Purpose: Implement creating an induced matrix distributions
# using approximation to the leverage scores of a matrix

"""
    approximate_leverage_score_distribution(A::AbstractMatrix, 
        row_sketching_method_FJLT::Function, col_sketching_method_JLT::Function,
        row_distribution::Bool)

Compute a distribution over the rows or columns of a matrix `A` as indicated by
`row_distribution` using approximate leverage scores.

# Reference(s)

Drineas, Magdon-Ismail, Mahoney, and Woodruff. "Fast approximation of matrix
coherence and statistical leverage". arxiv, https://arxiv.org/pdf/1109.3843.

# Method

Let `row_distribution = true` and ``A \\in \\mathbb{R}^{n \\times d}``.
Let ``\\Pi_1 \\in \\mathbb{R}^{r_1 \\times n}`` be a ``\\epsilon-``Fast Johnson-Lindenstrauss Transform, and
``\\Pi_2 \\in \\mathbb{R}^{d \\times r_2}`` be an ``\\epsilon-``Johnson-Lindenstrauss Transform. 

!!! note
    In the reference, ``\\Pi_1`` is formed by the subsampled randomized hadamard transform and an example distribution for ``\\Pi_2`` is introduced in 
    section 2.2.

Let the SVD of ``\\Pi_1A = U \\Sigma V^\\intercal`` and let 
``R^{-1} = V \\Sigma^{-1}``. Then, define

```math
    \\Omega = AR^{-1}\\Pi_2,
```
which is the matrix that will be used to construct distribution.

In particular, given ``\\Omega`` the probability weight assigned to the ``i^{th}``
row of ``A`` is

```math
    ||\\Omega[i, :]||_2^2/||\\Omega||_F^2.
```

If `row_distribution = false`, the same computation is carried out on 
``A^\\intercal``. The sketching methods passed into the function need to
account for this.

# Arguments

- `A::AbstractMatrix`, matrix for which a distribution over rows or columns is
    desired.
- `sketching_method_FJLT::Function`, sketching method that can be applied to the
    left of `A` if `row_distribution = true`, or to `A'` when 
    `row_distribution = false`. The function should take in a single argument,
    the matrix to be sketched, and return the sketched matrix.
- `sketching_method_JLT::Function`, sketching method that can applied to the
    right of a ``AR^{-1}`` when `row_distribution = true`, or to 
    ``A^{\\intercal}R^{-1}`` when `row_distribution = false`. The function 
    should take in a single argument, the matrix to be sketched, 
    and return the sketched matrix.
- `row_distribution::Bool`, whether a row or column distribution is desired.

# Return

- `distribution::Vector{Float64}`, vector of probability weights. Will be of
    size `size(A, 1)` when `row_distribution = true`, and of size `size(A, 2)`
    when `row_distribution = false`.
"""
function approximate_leverage_score_distribution(
    A::AbstractMatrix,
    sketching_method_FJLT::Function,
    sketching_method_JLT::Function,
    row_distribution::Bool
)

    B = row_distribution ? A : A'
    Ω = _approximate_leverage_score(B, sketching_method_FJLT, sketching_method_JLT)

    sz = size(B, 1)
    distribution = zeros(sz)
    for i in 1:sz
        distribution[i] = norm(Ω[i, :])^2
    end

    return distribution ./ sum(distribution)
end

"""
    _approximate_leverage_score(A::AbstractMatrix, row_sketching_method_FJLT::Function,
        col_sketching_method_JLT::Function)

Return the matrix for which the approximate leverage scores are computed from.

# Reference(s)

Drineas, Magdon-Ismail, Mahoney, and Woodruff. "Fast approximation of matrix
coherence and statistical leverage". arxiv, https://arxiv.org/pdf/1109.3843.

Let ``A \\in \\mathbb{R}^{n \\times d}``.
Let ``\\Pi_1 \\in \\mathbb{R}^{r_1 \\times n}`` be a ``\\epsilon-``Fast Johnson-Lindenstrauss Transform, and
``\\Pi_2 \\in \\mathbb{R}^{d \\times r_2}`` be an ``\\epsilon-``Johnson-Lindenstrauss Transform. 

!!! note
    In the reference, ``\\Pi_1`` is formed by the subsampled randomized hadamard transform and an example distribution for ``\\Pi_2`` is introduced in 
    section 2.2.

Let the SVD of ``\\Pi_1A = U \\Sigma V^\\intercal`` and let 
``R^{-1} = V \\Sigma^{-1}``. Then, define

```math
    \\Omega = AR^{-1}\\Pi_2.
```
This matrix is returned by the function, and use used to calculate the approximate 
leverage scores.

# Arguments

- `A::AbstractMatrix`, matrix for which a distribution over rows or columns is
    desired.
- `sketching_method_FJLT::Function`, sketching method that can be applied to the
    left of `A` if `row_distribution = true`, or to `A'` when 
    `row_distribution = false`. The function should take in a single argument,
    the matrix to be sketched, and return the sketched matrix.
- `sketching_method_JLT::Function`, sketching method that can applied to the
    right of a ``AR^{-1}`` when `row_distribution = true`, or to 
    ``A^{\\intercal}R^{-1}`` when `row_distribution = false`. The function should take 
    in a single argument, the matrix to be sketched, and return the sketched matrix.
- `row_distribution::Bool`, whether a row or column distribution is desired.

!!! warning
    The function `sketching_method_FLJT` should return a sketched matrix
    such that the dimension of the rows is at least the column dimension.
"""
function _approximate_leverage_score(
    A::AbstractMatrix,
    row_sketching_method_FJLT::Function,
    col_sketching_method_JLT::Function,
)

    # sketch the rows of the matrix A
    Π_1A = row_sketching_method_FJLT(A)

    # perform a QR decomposition
    _, Σ, V = svd(Π_1A)
    Rinv = V * Diagonal(Σ.^(-1))

    # sketch columns
    Ω = col_sketching_method_JLT(A * Rinv)
    return Ω
end