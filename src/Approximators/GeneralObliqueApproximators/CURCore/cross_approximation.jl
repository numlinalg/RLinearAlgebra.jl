"""
    CrossApproximation <: Core

A form of the core matrix in CUR decomposition, such that for column indices, J, and row
indices, I, the matrix ``U`` is ``A[I,J]^\\dagger``.

# Fields
- None
"""
struct CrossApproximation <: Core end

"""
    CrossApproximationRecipe <: Core

A structure featuring the preallocated arays necessary for storing the cross-approximation
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
mutable struct CrossApproximation <: CoreRecipe
    n_rows::Int64
    n_cols::Int64
    core::AbstractMatrix
    core_view::SubArray
    qr_decomp::QR
end

function complete_core(decomp::CUR, core::CrossApproximation,  A::AbstractMatrix)
    # preallocate the core matrix with and Idenity of the same type. We use identy because
    # this form preallocates all dense entries and works for sparse arrays.
    core = typeof(A)(I, decomp.n_rows, decomp.n_cols)
    qr_decomp = qr!(view(core, 1:2, 1:2))
    return CrossApproximation(decomp.n_rows, decomp.n_cols, core, qr_decomp)
end
 
function update_core!(core::CrossApproximationRecipe, decomp::CURRecipe, A::AbstractMatrix)
    # set the core to be the number of rows and columns in the matrix
    core.core_view = view(core.core, 1:decomp.n_rows, 1:decomp.n_cols)
    copyto!(core.core_view, A(decomp.row_idx, decomp.col_idx))
    core.qr_decomp = qr!(core.core_view)
    return nothing
end

function mul!(
    C::AbstractArray, 
    core::CrossApproximationRecipe, 
    A::AbstractArray,
    alpha::Number,
    beta::Number
)
    # to apply ldiv in place of 5 arg mul we first scale C by beta / alpha then apply alpha 
    # at the end: alpha A + beta C = alpha (A + beta/alpha C)
    lmul!(beta / alpha, C)
    ldiv!(C, core.qr_decomp, A)
    lmul!(alpha, C)
    return nothing
end
    
