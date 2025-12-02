"""
    Core

An abstract type for the computation of the core linking matrix in a CUR decomposition.
"""
abstract type Core end

"""
    CoreRecipe

An abstract type for the recipes containg the preallocated information needed for 
the computation of the core linking matrix in a CUR decomposition.
"""
abstract type CoreRecipe end

"""
    CUR

A struct that implements the CUR decomposition for forming low-rank approximations to 
    a matrix, ``A``. This technique selects column subsets, ``C``, row subsets, ``R``, and a 
    linking matrix, ``U``(this can either be ``C^\\dagger A R^\\dagger`` or ``A[I,J]``, 
    where ``I`` is a set of row indices and ``J`` is a set of column indices) such that 
    ``
        A \\approx CUR.
    ``
This will always form the CUR approximation by selecting columns followed by rows. If you 
    desire to approximate columns first, input `A'`.
# Mathamatical Description
In general finding a set of ``I`` and ``J`` that minimize the qpproximation quality is a
    NP-hard problem (you're not going to do it); thus, we often aim to find sufficiently 
    good indices. The best known quality of a rank ``r`` cur approximation to a  matrix
    ``\\tilde A_r`` is known to be 
    ``
        \\|A - \\tilde A_r \\|_F \\leq (r + 1) \|A - A_r\|_F,
    ``
    where ``A_r`` is the rank-``r`` truncated svd.

In practice numerous randomized methods match the performance of this best possible 
    selection procedure. These approaches can be broken into sampling and pivoting 
    approaches. The sampling approaches, like leverage score sampling, typically 
    have better theory, and involve imputed a distribution over the rows/columns of a matrix
    and drawing from that distribution with replacement. Randomized pivoting procedures
    on the other hand tend to be more efficient and accurate in practice and involve 
    compressing the matrix then applying a pivoting procedure on the compressed form.

# Fields
- `rank::Int64`, the desired rank of approximation.
- `oversample::Int64`, the amount of extra indices we wish to select with the row selection
    procedure. By default this is zero, although it can improve the stability of the cross
    approximation core.
- `col_selector::Selector`, the technique used for selecting column indices from a matrix.
- `row_selector::Selector`, the technique used for selecting row indices from a matrix.
- `core::Core`, the method for computing the core linking matrix, `U`, in the CUR.

# Constructor
    CUR(rank;
        oversample = 0,
        selector_cols = LUPP(),
        selector_rows = selector_cols,
        core = CrossApproximation(),
    )
"""

mutable struct CUR
    rank::Int64
    oversample::Int64
    col_selector::Selector
    row_selector::Selector
    core::Core
end


function CUR(rank;
        oversample = 0,
        selector_cols = LUPP(),
        selector_rows = selector_cols,
        core = CrossApproximation(),
    )
    return CUR(rank, oversample, selector_cols, selector_rows, core)
end

"""
    CURRecipe

A struct that contains the preallocated memory, completed compressor, and selector to form
    a CUR approximation.
"""
mutable struct CURRecipe{CR<:CoreRecipe}
    n_rows::Int64
    n_cols::Int64
    row_idx::Vector{Int64}
    col_idx::Vector{Int64}
    C::AbstractMatrix
    U::CR
    R::AbstractMatrix
end

function complete_approximator(ingredients::CUR, A::AbstractMatrix)
    n_col_vecs = ingredients.rank
    n_row_vecs = ingredients.rank + ingredients.oversample
    col_idx = zeros(Int64, n_col_vecs)
    row_idx = zeros(Int64, n_row_vecs)
    col_selector = complete_selector(ingredients.col_selector, A)
    row_selector = complete_selector(ingredients.row_selector, A)
    C = Matrix{eltype(A)}(undef, size(A, 1), n_col_vecs)
    R = Matrix{eltype(A)}(undef, n_row_vecs, size(A, 2))
    U = complete_core(CUR, CUR.core, A)
    return CURRecipe(n_row_vecs, n_col_vecs, row_idx, col_idx, C, U, R)
end

function rapproximate!(appprox::CURRecipe{CrossApproximationRecipe}, A::AbstractMatrix)
    # select column indices
    select_indices!(
        approx.col_idx,
        approx.col_selector,
        A,
        approx.n_cols,
        1
    )
    
    # gather the columns to select rows dependently 
    copyto!(approx.C, A[:, approx.col_idx])

    # select row indices
    select_indices!(
        approx.row_idx,
        approx.row_selector,
        approx.C',
        approx.n_rows,
        1
    )

    # gather the rows entries 
    copyto!(approx.R, A[:, approx.row_idx])
    # Compute the core matrix
    update_core!(approx.U, approx, A)
    return nothing
end

function rapproximate!(appprox::CURRecipe{CrossApproximationRecipe}, A::AbstractMatrix)
    # select column indices
    select_indices!(
        approx.col_idx,
        approx.col_selector,
        A,
        approx.n_cols,
        1
    )
    

    # select row indices
    select_indices!(
        approx.row_idx,
        approx.row_selector,
        A',
        approx.n_rows,
        1
    )

    # gather the rows entries 
    copyto!(approx.R, A[:, approx.row_idx])
    # gather column entries
    copyto!(approx.C, A[:, approx.col_idx])
    # Compute the core matrix
    update_core!(approx.U, approx, A)
    return nothing
end

function rapproximate(approx::CUR, A::AbstractMatrix)
    approx_recipe = complete_approximator(approx, A)
    rapproximate!(approx_recipe, A)
    return  approx_recipe
end

# Implement the muls
