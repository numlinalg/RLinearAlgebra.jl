"""
    RangeFinder

A struct that implements the Randomized Range Finder technique which uses compression from 
    the right to form an low-dimensional orthogonal matrix ``Q`` that approximates the 
    range of ``A``. See [halko2011finding](@cite) for additional details.

# Mathematical Description
Suppose we have a matrix ``A \\in \\mathbb{R}^{m \\times n}`` of which we wish to form a low 
    rank approximation that approximately captures the range of ``A``. Specifically, we wish
    to find an Orthogonal matrix ``Q`` such that ``QQ^\\top A \\approx A``. 

    A simple way to find such a matrix is to choose a ``k`` representing the number of 
    vectors we wish to have in the subspace. Then we can generate a compression matrix 
    ``S\\in\\mathbb{R}^{n \\times k}`` and compute ``Q = \\text{qr}(AS)``. 
    With high probability we will have ``\\|A - QQ^\\top A\\|_2 \\leq
    (k+1) \\sigma_{k+1}``, where ``\\sigma_{k+1}`` is the ``k+1^\\text{th}`` singular value 
    of A. This bound is often conservative when the singular values of ``A`` decay quickly. 
    In the case where the singular values decay slowly, by computing the qr factorization of
    ``(AA^\\top)^q AS``, this is known as taking ``q`` power iterations. Power iterations 
    drive the ``k+1`` constant in front of ``\\sigma_{k+1}`` in the bound closer to 1, 
    leading to more accurate approximations. One can also improve the stability of these 
    power iterations be orthogonalizing each matrix in what is known as the random subspace 
    iteration.

# Fields
- `compressor::Compressor`, the technique that will compress the matrix from the right.
- `power_its::Int64`, the number of power iterations that should be performed.
- `rand_subspace::Bool`, a boolean indicating whether the `power_its` should be performed 
    with orthogonalization.
"""
mutable struct RangeFinder <: RangeApproximator
    compressor::Compressor
    power_its::Int64
    rand_subspace::Bool
    function RangeFinder(compressor, power_its, rand_subspace)
        if power_its < 0
            return throw(ArgumentError("Field `power_its` must be non-negative."))
        end
        
        return new(compressor, power_its, rand_subspace)
    end

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
- `rand_subspace::Bool`, a boolean indicating whether the `power_its` should be performed 
    with orthogonalization.
- `range::AbstractMatrix`, the orthogonal matrix that approximates the range of ``A``.
"""
mutable struct RangeFinderRecipe <: RangeApproximatorRecipe
    n_rows::Int64
    n_cols::Int64
    compressor::CompressorRecipe
    power_its::Int64
    rand_subspace::Bool
    range::AbstractMatrix
end

function complete_approximator(approx::RangeFinder, A::AbstractMatrix)
    type = eltype(A)
    # You need to make sure you orient the compressor in the correct direction
    if typeof(approx.compressor.cardinality) <: Left
        @warn "Compressor with cardinality `Left` being applied from `Right`."
    end

    compress = complete_compressor(approx.compressor, A)
    # Determine the dimensions of the range approximator
    a_rows = size(A, 1)
    c_cols = size(compress, 2)
    approx_recipe = RangeFinderRecipe(
        a_rows,
        c_cols,
        compress, 
        approx.power_its,
        approx.rand_subspace, 
        Matrix{type}(undef, 2, 2)
    )
end

function rapproximate!(approx::RangeFinderRecipe, A::AbstractMatrix)
    # User may wish to choose to use a different subspace iteration
    if approx.rand_subspace 
        approx.range = rand_power_it(A, approx)
    else
        approx.range = rand_subspace_it(A, approx)
    end

    return nothing
end

function rapproximate(approx::RangeFinder, A::AbstractMatrix)
    approx_recipe = complete_approximator(approx, A)
    rapproximate!(approx_recipe, A)
    return  approx_recipe
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
    mul!(C, R.parent.range', A, alpha, beta)
end

function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    R::RangeFinderRecipe, 
    alpha::Number, 
    beta::Number
)
    mul!(C, A, R.range, alpha, beta)
end


function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    R::ApproximatorAdjoint{RangeFinderRecipe}, 
    alpha::Number, 
    beta::Number
)
    mul!(C, A, R.parent.range', alpha, beta)
end
