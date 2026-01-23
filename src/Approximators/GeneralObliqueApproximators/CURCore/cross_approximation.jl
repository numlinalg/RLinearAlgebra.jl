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
- `core_view::SubArray`, a sub-matrix for the part of the allocated core where the actual 
    values of the core are stored. This allows you to increase the size of the core matrix 
    without additional memory allocations.
- `qr_decomp::QR`, storage for the inplace QR decomposition of the core.
"""
mutable struct CrossApproximationRecipe <: CURCoreRecipe
    n_rows::Int64
    n_cols::Int64
    core::AbstractMatrix
    core_view::SubArray
    qr_decomp::Union{QRCompactWY, SPQR.QRSparse}
end

function complete_core(cur::CUR, core::CrossApproximation,  A::AbstractMatrix)
    # preallocate the core matrix with and Identity of the same type. We use identy because
    # this form preallocates all dense entries and works for sparse arrays.
    core = typeof(A)(I, cur.rank + cur.oversample, cur.rank)
    core_view = view(core, 1:2, 1:2)
    qr_decomp = qr!(core_view)
    return CrossApproximationRecipe(
        cur.rank, 
        cur.rank + cur.oversample, 
        core, 
        core_view, 
        qr_decomp
    )
end
 
function update_core!(core::CrossApproximationRecipe, cur::CURRecipe, A::AbstractMatrix)
    # set the core to be the number of rows and columns in the matrix
    # the core view is the number of rows by number of columns even though the size
    # the core is number of columns by number of rows because of the transpose from the 
    # implicit inversion of the core matrix
    core.core_view = view(core.core, 1:cur.n_row_vecs, 1:cur.n_col_vecs)
    copyto!(core.core_view, A[cur.row_idx, cur.col_idx])
    core.qr_cur = qr!(core.core_view)
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
    E = deepcopy(C)
    lmul!(beta, E)
    ldiv!(C, core.qr_decomp, A)
    lmul!(alpha, C)
    C .+= E
    return nothing
end
    
function mul!(
    C::AbstractArray, 
    core::CURCoreAdjoint{CrossApproximationRecipe}, 
    A::AbstractArray,
    alpha::Number,
    beta::Number
)
    E = typeof(A)(undef, size(core, 2), size(A, 2)) 
    ldiv!(E, LowerTriangular(core.parent.qr_decomp.R'), A)
    mul!(C, Array(core.parent.qr_decomp.Q), E, alpha, beta)
    return nothing
end

function mul!(
    C::AbstractArray, 
    A::AbstractArray,
    core::CrossApproximationRecipe, 
    alpha::Number,
    beta::Number
)
    E = typeof(A)(undef, size(A, 1), size(core, 1))
    #AR^(-1) = (R^(-1)' A')' = ((R')^(-1)A')'
    ldiv!(E', LowerTriangular(core.qr_decomp.R'), A')
    # then multiply the Q from the right
    mul!(C, E, core.qr_decomp.Q', alpha, beta)
    return nothing
end

function mul!(
    C::AbstractArray, 
    A::AbstractArray,
    core::CURCoreAdjoint{CrossApproximationRecipe}, 
    alpha::Number,
    beta::Number
)
    E = typeof(A)(undef, size(core, 1), size(core, 2))
    # form pseudoinverse matrix
    ldiv!(E', core.qr_decomp.R, core.qr_decomp.Q')
    # then multiply the E from the right
    mul!(C, A, E', alpha, beta)
    return nothing    
end