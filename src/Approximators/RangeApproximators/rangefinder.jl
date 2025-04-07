"""
    RangeFinder

A struct that implements the Randomized Range Finder technique which uses compression from 
    the right to form an low-dimensional orthogonal matrix ``Q`` thar approximates the 
    range of ``A``. See [](@cite) for additional details.

# Mathematical Description
Suppose we have a matrix ``A \\in \\mathbb{R}^{m \\times n}`` of which we wish to form a low 
    rank approximation that approximately captures the range of ``A``. Specifially, we wish
    to find an Orthogonal matrix ``Q`` such that ``QQ^\\top A \\approx A``. A simple way to
    this is is to choose a ``k`` representing the number of columns we wish to have in the
    subspace and generatign a compression matrix ``S\\in\\mathbb{R}^{n \\times k}``. Then by
    computing ``Q = \\text{qr}(AS)``, with high probability ``\\|A - QQ^\\top A\\|_2 \\leq
    ``(k+1) \\sigma_{k+1}`` where ``\\sigma_{k+1}`` is the ``k+1^\\text{th}`` singular value 
    of A. This bound is often too loose when the singular values of ``A`` decay quickly. 
    In the case where the singular values decay slowly, by computing the qr factorization of
    ``(AA^\\top)^q AS``, this is known as taking ``q`` power iterations. Power iterations 
    drive the constant on ``\\sigma_{k+1}`` in the bound closer to 1, leading to more 
    accurate approximations. One can also improve the stability of these power iterations
    be orthogonalizing each matrix in what is known as the random subspace iteration.

# Fields
- `compressor::Compressor`, the technique that will compress the matrix from the right.
- `power_its::Int64`, the number of power iterations that should be performed.
- `rand_subspace::Bool`, a boolean indicating whether the power its should be performed 
    with orthogonalization.
"""
mutable struct RangeFinder
    compressor::Compressor
    power_its::Int64
    rand_subspace::Bool
end

RangeFinder(;
    compressor = SparseSign(), 
    rand_subspace = false, 
    power_its = 1
   ) = RangeFinder(compressor, rand_subspace, power_its)

"""
    RangeFinderRecipe

A struct that contains the preallocated memory and completed compressor to form a
    RangeFinder approximation to the matrix ``A``.

# Fields
- `compressor::CompressorRecipe`, the compressor to be applied from the right to ``A``.
- `power_its::Int64`, the number of power iterations that should be performed.
- `rand_subspace::Bool`, a boolean indicating whether the power its should be performed 
    with orthogonalization.
- `range::AbstractMatrix`, the orthogonal matrix that approximates the range of ``A``.
"""
mutable struct RangeFinderRecipe
    compressor::CompressorRecipe
    power_its::Int64
    rand_subspace::Bool
    range::Union{Nothing, AbstractMatrix}
end

function rapproximate!(approx::RangeFinderRecipe, A::AbstractMatrix)
    # we don't dispatch on this incase someone wishes to make multiple runs with the 
    # same recipe
    if rand_subspace 
        approx.range = rand_power_it(A, approx)
    else
        approx.range = rand_subspace_it(A, approx)
    end

    return nothing
end

function rapproximate(approx::RangeFinder, A::AbstractMatrix)
    type = eltype(A)
    # You need to make sure you orient the compressor in the correct direction
    if approx.compress.carindality == Left
        approx.compress.carindality = Right
    end
    compress = complete_compressor(approx.compressor, A)
    
    approx_recipe = RangeFinderRecipe(
        compress, 
        approx.rand_subspace, 
        approx.power_its,
        Matrix{type}(undef, 2, 2)
    )

    rapproximate!(approx_recipe, A)

    return nothing
end

function mul!(
    C::AbstractArray, 
    R::RangeFinderRecipe, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    mul!(C, R.range, A, alpha, beta)
end


function mul!(
    C::AbstractArray, 
    R::ApproximatorAdjoint{RangeFinderRecipe}, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    mul!(C, R.range', A, alpha, beta)
end
