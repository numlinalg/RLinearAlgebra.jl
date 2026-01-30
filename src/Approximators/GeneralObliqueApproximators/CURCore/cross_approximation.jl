"""
    CrossApproximation <: CURCore

A form of the core matrix in CUR decomposition, such that for column indices, J, and row
indices, I, the matrix ``U`` is ``A[I,J]^\\dagger``.

# Fields
- None
"""
struct CrossApproximation <: CURCore end

"""
    CrossApproximationRecipe <: CURCore

A structure featuring the preallocated arrays necessary for storing the cross-approximation
core matrix to a CUR decomposition.

# Fields
- `n_rows::Int64`, the number of rows in the core matrix.
- `n_cols::Int64`, the number of columns in the core matrix.
- `core::AbstractMatrix`, the core matrix, found as the intersection of the row and column
    indices.
"""
mutable struct CrossApproximationRecipe <: CURCoreRecipe
    n_rows::Int64
    n_cols::Int64
    core::AbstractMatrix
end

function complete_core(core::CrossApproximation, cur::CUR,  A::AbstractMatrix)
    # preallocate the core matrix with and Identity of the same type. We use identy because
    # this form preallocates all dense entries and works for sparse arrays.
    core = typeof(A)(I, cur.rank + cur.oversample, cur.rank)
    return CrossApproximationRecipe(
        cur.rank, 
        cur.rank + cur.oversample, 
        core 
    )
end
 
function update_core!(core::CrossApproximationRecipe, cur::CURRecipe, A::AbstractMatrix)
    # set the core to be the number of rows and columns in the matrix
    # the core view is the number of rows by number of columns even though the size
    # the core is number of columns by number of rows because of the transpose from the 
    # implicit inversion of the core matrix
    Q, R = qr!(A[cur.row_idx, cur.col_idx])
    core.core = R \ Array(Q)'
    return nothing
end


# Implement the rapproximate function for the Cross Approximation 
function rapproximate!(approx::CURRecipe{CrossApproximationRecipe}, A::AbstractMatrix)
    # select column indices
    select_indices!(
        approx.col_idx,
        approx.col_selector,
        A,
        approx.n_col_vecs,
        1
    )
    
    # gather the columns to select rows dependently 
    copyto!(approx.C, A[:, approx.col_idx])

    # select row indices depent on the columns selected
    select_indices!(
        approx.row_idx,
        approx.row_selector,
        approx.C',
        approx.n_row_vecs,
        1
    )

    # gather the rows entries 
    copyto!(approx.R, A[approx.row_idx, :])
    # Compute the core matrix
    update_core!(approx.U, approx, A)
    return nothing
end

# you still need to figure out how to do these multiplications now the issue is that type of (A) does not work when A is a vector
function mul!(
    C::AbstractArray, 
    core::CrossApproximationRecipe, 
    A::AbstractArray,
    alpha::Number,
    beta::Number
)
    mul!(C, core.core, A, alpha, beta)
    return nothing
end
    
function mul!(
    C::AbstractArray, 
    A::AbstractArray,
    core::CrossApproximationRecipe, 
    alpha::Number,
    beta::Number
)
    mul!(C, A, core.core, alpha, beta)
    return nothing
end