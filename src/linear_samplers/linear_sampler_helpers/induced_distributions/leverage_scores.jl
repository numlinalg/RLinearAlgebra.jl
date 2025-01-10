# Date: 12/19/2024
# Author: Christian Varner
# Purpose: Create functionality for distributions for rows and columns based
# on the leverage scores.

"""
    leverage_score_distribution(A::AbstractMatrix, row_distribution::Bool)

Given a matrix `A`, compute a distribution over rows or columns of the matrix, as 
    indicated by `row_distribution`, by using the leverage scores of the matrix.

# References(s)

Drineas, Magdon-Ismail, Mahoney, and Woodruff. "Fast Approximation of Matrix Coherence
and Statistical Leverage". arxiv, https://arxiv.org/abs/1109.3843v2.

# Method

If `row_distribution = true`, then we compute the thin QR decomposition of `A`.
Take ``Q_1`` to be the thin Q. The probability weight assigned to row ``i`` of `A` is
```math
    ||Q_1[i, :]||_2^2 / ||Q_1||_F^2,
```
where ``Q_1[i, :]`` is the ``i^{th}`` row of ``Q_1``, ``||\\cdot||_2`` is the L2 norm, and
``||\\cdot||_F`` is the frobenius norm.

If `row_distribution = false`, then we carry out the same procedure on ``A^\\intercal``.

# Arguments

- `A::AbstractMatrix`, matrix for which a distribution over rows or columns is desired.
- `row_distribution::Bool`, indication whether a distribution over rows or columns
    is desired.

!!! note
    As we take the QR decomposition of `A`, we enforce that `size(A)[1] >= size(A)[2]` 
    when `row_distribution = true`, and `size(A')[1] >= size(A')[2]` when 
    `row_distribution = false`.

# Return

- `distribution::Vector{Float64}`, vector of probabilities. Will be of length `size(A, 1)` 
    if `row_distribution = true`, otherwise will be of length `size(A, 2)`.

!!! warning
    If `A` is sparse, the implementation does not account for this.
    The distribution vector is always initialized to be of type `Vector{Float64}`.
"""
function leverage_score_distribution(
    A::AbstractMatrix,
    row_distribution::Bool
)
    # error checking
    @assert row_distribution ? (size(A)[1] >= size(A)[2]) : (size(A')[1] >= size(A')[2])
    "Need to take QR decomposition but dimensions of A are incorrect."

    # get thin QR
    sz = row_distribution ? size(A)[1] : size(A)[2]
    distribution = zeros(sz)
    Q1 = row_distribution ? Matrix(qr(A).Q) : Matrix(qr(A').Q)
    
    # compute distribution
    for i in 1:sz
        distribution[i] = row_distribution ? norm(view(Q1, i, :))^2 : norm(view(Q1, i, :))^2
    end

    return distribution ./ sum(distribution)
end