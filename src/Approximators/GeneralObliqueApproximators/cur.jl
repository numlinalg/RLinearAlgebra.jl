###########################################
# Include the CURCore definition
##########################################
include("CURCore.jl")

"""
    CUR <: Approximator

A struct that implements the CUR decomposition for forming low-rank approximations to 
    a matrix, ``A``. This technique selects column subsets, ``C``, row subsets, ``R``, and a 
    linking matrix, ``U``(this can either be ``C^\\dagger A R^\\dagger`` or ``A[I,J]``, 
    where ``I`` is a set of row indices and ``J`` is a set of column indices) such that 
    ``
        A \\approx CUR.
    ``
This will always form the CUR approximation by selecting columns followed by rows. If you 
    desire to approximate columns first, input `A'`.
# Mathematical Description
In general finding a set of ``I`` and ``J`` that minimize the qpproximation quality is a
    NP-hard problem (you're not going to do it); thus, we often aim to find sufficiently 
    good indices. The best known quality of a rank ``r`` cur approximation to a  matrix
    ``\\tilde A_r`` is known to be 
    ``
        \\|A - \\tilde A_r \\|_F \\leq (r + 1) \\|A - A_r\\|_F,
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
mutable struct CUR <: Approximator
    rank::Int64
    oversample::Int64
    col_selector::Selector
    row_selector::Selector
    core::CURCore
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
    CURRecipe <: ApproximatorRecipe

A struct that contains the CUR approximation. It is created  
"""
mutable struct CURRecipe{CR<:CURCoreRecipe} <: ApproximatorRecipe
    n_row_vecs::Int64
    n_col_vecs::Int64
    col_selector::SelectorRecipe
    row_selector::SelectorRecipe
    row_idx::Vector{Int64}
    col_idx::Vector{Int64}
    C::AbstractMatrix
    U::CR
    R::AbstractMatrix
    buffer_row::AbstractArray
    buffer_core::AbstractArray
end


###############################################
# Include files for core implementations
###############################################
include("./CURCore/cross_approximation.jl")

# write the size functions for CUR
function size(approx::CURRecipe)
    return size(approx.C, 1), size(approx.R, 2)
end

function Base.size(S::CURRecipe, dim::Int64)
    ((dim < 1) || (dim > 2)) && throw(DomainError("`dim` must be 1 or 2."))
    return dim == 1 ? size(approx.C, 1) : size(approx.R, 2)
end

function size(approx::Adjoint{CURRecipe})
    return size(approx.R, 2), size(approx.C, 1)
end

function Base.size(S::Adjoint{CURRecipe}, dim::Int64)
    ((dim < 1) || (dim > 2)) && throw(DomainError("`dim` must be 1 or 2."))
    return dim == 1 ? size(approx.R, 2) : size(approx.C, 1)
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
        col_selector,
        row_selector,
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

# implenetations of rapproximate can be found with the core matrix implementations
# This is done to ensure stability as different core matrix approaches require 
# different organizations of the CUR selection

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
