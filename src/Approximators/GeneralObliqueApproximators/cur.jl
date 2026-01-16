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

function size(approx::CoreRecipe)
    return approx.n_rows, approx.n_cols
end

function size(approx::CoreRecipe, 1)
    return approx.n_rows
end

function size(approx::CoreRecipe, 2)
    return approx.n_cols
end

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
- `blocksize::Int64`, number of vectors stored in a buffer matrix for multiplication.
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
    blocksize::Int64
end


function CUR(rank;
        oversample = 0,
        selector_cols = LUPP(),
        selector_rows = selector_cols,
        core = CrossApproximation(),
    )
    return CUR(rank, oversample, selector_cols, selector_rows, core, 0)
end

"""
    CURRecipe

A struct that contains the preallocated memory, completed compressor, and selector to form
    a CUR approximation.
"""
mutable struct CURRecipe{CR<:CoreRecipe}
    n_row_vecs::Int64
    n_col_vecs::Int64
    row_idx::Vector{Int64}
    col_idx::Vector{Int64}
    C::AbstractMatrix
    U::CR
    R::AbstractMatrix
    buffer_row::AbstractArray
    buffer_core::AbstractArray
end

# write the size functions for CUR
function size(approx::CURRecipe)
    return (size(approx.C, 1), size(approx.R, 2))
end

function size(approx::CURRecipe, 1)
    return size(approx.C, 1)
end

function size(approx::CURRecipe, 2)
    return size(approx.R, 2)
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
    return CURRecipe(
        n_row_vecs, 
        n_col_vecs, 
        row_idx, 
        col_idx, 
        C, 
        U, 
        R, 
        # because of oversampling
        zeros(n_row_vecs, ingredients.blocksize),
        zeros(n_col_vecs, ingredients.blocksize)
    )
end

# add function for computing the size of the CURRecipe
function size(A::CURRecipe)
    return size(A.C, 1), size(A.R, 2)
end


function size(A::CURRecipe, 2)
    return size(A.R, 2)
end

function size(A::CURRecipe, 1)
    return size(A.C, 1)
end

# Implement the rapproximate function
function rapproximate!(appprox::CURRecipe{CrossApproximationRecipe}, A::AbstractMatrix)
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

    # select row indices
    select_indices!(
        approx.row_idx,
        approx.row_selector,
        approx.C',
        approx.n_row_vecs,
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
        approx.n_col_vecs,
        1
    )
    

    # select row indices
    select_indices!(
        approx.row_idx,
        approx.row_selector,
        A',
        approx.n_row_vecs,
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
function mul!(
    C::AbstractArray, 
    approx::CURRecipe, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # determine size of buffer we have and how much we need
    # okay just checking row buffer because both have the same number of columns
    buff_r_n = size(approx.buffer_row, 2)
    m, n = size(A)
    n_its = div(n, buff_r_n)
    last_block = rem(n, buff_r_n)
    # if the buffer is bigger than necessary set blocksize to be the 
    # number of columns in A
    blocksize = min(buff_r_n, n)
    # Perform first iteration to scale C by beta
    br_v = view(approx.buffer_row, :, 1:blocksize)
    bc_v = view(approx.buffer_core, :, 1:blocksize)
    Cv = view(C, :, 1:blocksize)
    mul!(br_v, approx.R, A[:, 1:blocksize])
    mul!(bc_v, approx.U, br_v)
    mul!(Cv, approx.C, bc_v, alpha, beta) 

    start = blocksize + 1
    # perform remaining necessary iterations
    for i = 2:n_its
        last = start + blocksize
        # if in the loop using all columns of buffer
        br_v = view(approx.buffer_row, :, :)
        bc_v = view(approx.buffer_core, :, :)
        Cv = view(C, :, start:last)
        mul!(br_v, approx.R, A[:, start:last])
        mul!(bc_v, approx.U, br)
        mul!(Cv, approx.C, bc_v, alpha, beta)
        start = last + 1
    end

    # complete the last block which could differ from block size
    if n_its > 0 && last_block > 0
        last = start + last_block 
        # perform last block
        br_v = view(approx.buffer_row, :, 1:last_block)
        bc_v = view(approx.buffer_core, :, 1:last_block)
        Cv = view(C, :, start:last)
        mul!(br_v, approx.R, A[:, start:last])
        mul!(bc_v, approx.U, br_v)
        mul!(Cv, approx.C, bc_v, alpha, beta)
    end

    return
end

function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    approx::CURRecipe, 
    alpha::Number, 
    beta::Number
)
    # determine size of buffer we have and how much we need
    # okay just checking row buffer because both have the same number of columns
    buff_r_n = size(approx.buffer_row, 2)
    m, n = size(A)
    n_its = div(m, buff_r_n)
    last_block = rem(m, buff_r_n)
    # if the buffer is bigger than necessary set blocksize to be the 
    # number of columns in A
    blocksize = min(buff_r_n, n)
    # Perform first iteration to scale C by beta
    br_v = view(approx.buffer_row, :, 1:blocksize)
    bc_v = view(approx.buffer_core, :, 1:blocksize)
    Cv = view(C, 1:blocksize, :)
    mul!(bc_v', A[1:blocksize, :], approx.C)
    mul!(br_v', bc_v', approx.U)
    mul!(Cv, br_v', approx.R, alpha, beta) 

    start = blocksize + 1
    # perform remaining necessary iterations
    for i = 2:n_its
        last = start + blocksize
        # if in the loop using all columns of buffer
        br_v = view(approx.buffer_row, :, :)
        bc_v = view(approx.buffer_core, :, :)
        Cv = view(C, start:last, :) 
        mul!(bc_v', A[start:last, :], approx.C)
        mul!(br_v', bc_v', approx.U)
        mul!(Cv, br_v', approx.R, alpha, beta) 
        start = last + 1
    end

    # complete the last block which could differ from block size
    if n_its > 0 && last_block > 0
        last = start + last_block
        # perform last block
        br_v = view(approx.buffer_row, :, 1:last_block)
        bc_v = view(approx.buffer_core, :, 1:last_block)
        Cv = view(C, start:last, :)
        mul!(bc_v', A[start:last, :], approx.C)
        mul!(br_v', bc_v', approx.U)
        mul!(Cv, br_v', approx.R, alpha, beta) 
    end

    return
end

function mul!(
    C::AbstractArray, 
    approx::Adjoint{CURRecipe}, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    # determine size of buffer we have and how much we need
    # okay just checking row buffer because both have the same number of columns
    buff_r_n = size(approx.buffer_row, 2)
    m, n = size(A)
    n_its = div(n, buff_r_n)
    last_block = rem(n, buff_r_n)
    # if the buffer is bigger than necessary set blocksize to be the 
    # number of columns in A
    blocksize = min(buff_r_n, n)
    # Perform first iteration to scale C by beta
    br_v = view(approx.buffer_row, :, 1:blocksize)
    bc_v = view(approx.buffer_core, :, 1:blocksize)
    Cv = view(C, :, 1:blocksize)
    mul!(bc_v, approx.C', A[:, 1:blocksize])
    mul!(br_v, approx.U', bc_v)
    mul!(Cv, approx.R', br_v, alpha, beta) 

    start = blocksize + 1
    # perform remaining necessary iterations
    for i = 2:n_its
        last = start + blocksize
        # if in the loop using all columns of buffer
        br_v = view(approx.buffer_row, :, :)
        bc_v = view(approx.buffer_core, :, :)
        Cv = view(C, :, start:last)
        mul!(bc_v, approx.C', A[:, start:last])
        mul!(br_v, approx.U', bc_v)
        mul!(Cv, approx.R', br_v, alpha, beta) 
        start = last + 1
    end

    # complete the last block which could differ from block size
    if n_its > 0 && last_block > 0
        last = start + last_block 
        # perform last block
        br_v = view(approx.buffer_row, :, 1:last_block)
        bc_v = view(approx.buffer_core, :, 1:last_block)
        Cv = view(C, :, start:last)
        mul!(bc_v, approx.C', A[:, start:last])
        mul!(br_v, approx.U', bc_v)
        mul!(Cv, approx.R', br_v, alpha, beta) 
    end

    return
end

function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    approx::Adjoint{CURRecipe}, 
    alpha::Number, 
    beta::Number
)
    # determine size of buffer we have and how much we need
    # okay just checking row buffer because both have the same number of columns
    buff_r_n = size(approx.buffer_row, 2)
    m, n = size(A)
    n_its = div(m, buff_r_n)
    last_block = rem(m, buff_r_n)
    # if the buffer is bigger than necessary set blocksize to be the 
    # number of columns in A
    blocksize = min(buff_r_n, n)
    # Perform first iteration to scale C by beta
    br_v = view(approx.buffer_row, :, 1:blocksize)
    bc_v = view(approx.buffer_core, :, 1:blocksize)
    Cv = view(C, 1:blocksize, :)
    mul!(br_v', A[1:blocksize, :], approx.R')
    mul!(bc_v', br_v', approx.U')
    mul!(Cv, bc_v', approx.C', alpha, beta) 

    start = blocksize + 1
    # perform remaining necessary iterations
    for i = 2:n_its
        last = start + blocksize
        # if in the loop using all columns of buffer
        br_v = view(approx.buffer_row, :, :)
        bc_v = view(approx.buffer_core, :, :)
        Cv = view(C, start:last, :) 
        mul!(br_v', A[start:last, :], approx.R')
        mul!(bc_v', br_v', approx.U')
        mul!(Cv, bc_v', approx.C', alpha, beta) 
        start = last + 1
    end

    # complete the last block which could differ from block size
    if n_its > 0 && last_block > 0
        last = start + last_block
        # perform last block
        br_v = view(approx.buffer_row, :, 1:last_block)
        bc_v = view(approx.buffer_core, :, 1:last_block)
        Cv = view(C, start:last, :)
        mul!(br_v', A[start:last, :], approx.R')
        mul!(bc_v', br_v', approx.U')
        mul!(Cv, bc_v', approx.C', alpha, beta)
    end

    return
end