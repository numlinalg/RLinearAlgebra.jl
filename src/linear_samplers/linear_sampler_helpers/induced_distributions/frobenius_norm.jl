# Date: 12/13/2024
# Author: Christian Varner
# Purpose: Create a function that produces a distribution over the rows
# or columns, as indicated by the user, of a matrix A

"""
    frobenius_norm_distribution(A::AbstractMatrix, row_distribution::Bool)

Given a matrix `A` and an indication of whether a distribution over rows or columns
should be created in `row_distribution`, return a probability vector.

# Method

If `row_distribution = true`, `distribution[i]` will be the probability weight assigned
to row `i` of `A`. It is computed as
```math
   ||A_{i,\\cdot}||_2^2 / ||A||_F^2  
```
where ``A_{i, \\cdot}`` is the ``i^{th}`` row of `A`, ``||\\cdot||_2`` is the L2 norm, and
``||\\cdot||_F`` is the Frobenius norm.

If `row_distribution = false`, `distribution[i]` will be the probability weight assigned to
column `i` of `A`. It is computed by an analogous formula as above, however with the norm
squared of the rows of `A` replace by the norm squared of the columns of `A`. 

# Arguments

- `A::AbstractMatrix`, matrix for which a distribution over rows or columns is desired.
- `row_distribution::Bool`, indication whether a distribution over rows or columns
    is desired.

# Return

- `distribution::Vector{Float64}`, vector of probabilities. Will be of `size(A, 1)` if
    `row_distribution = true`, otherwise will be of size `size(A, 2)`.

!!! warning
    If `A` is sparse, the implementation does not account for this.
"""
function frobenius_norm_distribution(
    A::AbstractMatrix,
    row_distribution::Bool
)
    # get distribution vector and range of iteration
    max_index = size(A, 1) ? row_distribution : size(A, 2)
    distribution = zeros(max_index)
    for i in 1:max_index
        distribution[i] = norm(@view A[i, :])^2 ? row_distribution : norm(@view A[:, i])^2
    end

    return distribution ./ sum(distribution)
end