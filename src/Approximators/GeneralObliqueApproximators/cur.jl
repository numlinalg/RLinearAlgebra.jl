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
abstract type Core end


struct Optimal <: Core end

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
- `compressor::Compressor`, when a selector needs to compress a matrix, this is the 
    compressor to be used.
- `selector::Selector`, the technique used for selecting indices from a matrix.
- `core::Core`, the method for computing the core linking matrix, `U`, in the CUR.

# Constructor
    CUR(rank;
        oversample = 0,
        compressor = SparseSign(),
        selector = LUPP(),
        core = CrossApproximation(),
    )
"""

mutable struct CUR
    rank::Int64
    oversample::Int64
    compressor::Compressor
    selector::Selector
    core::Core
end

"""
    CURRecipe

A struct that contains the preallocated memory, completed compressor, and selector to form
    a CUR approximation.
"""
mutable struct CURRecipe
    n_rows::Int64
    n_cols::Int64
    row_idx::Vector{Int64}
    col_idx::Vector{Int64}
    C::AbstractMatrix
    U::CoreRecipe
    R::AbstractMatrix
end
