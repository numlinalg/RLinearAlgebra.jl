"""
    RandSVD 

A struct that implements the Randomized SVD. The Randomized SVD technique compresses a 
    matrix from the right to compute a rank ``k`` estimate to the truncated 
    SVD of a matrix ``A``. See Algorithm 5.1 in [halko2011finding](@cite) for additional 
    details.

# Mathematical Description
Suppose we have a matrix ``A \\in \\mathbb{R}^{m \\times n}`` for which we wish to form a 
    low-rank approximation with the form of an SVD. Specifically, we wish to find an 
    orthogonal matrix ``U``, a diagonal matrix ``S``, and an orthogonal matrix ``V`` 
    such that ``USV^\\top \\approx A``. A simple way to find such a matrix is to choose a 
    ``k`` representing the number of singular vectors and values we wish to approximate. 
    With this ``k``,  we  generate a compression matrix 
    ``S\\in\\mathbb{R}^{n \\times k}`` and compute ``Q = \\text{qr}(AS)`` as in the 
    [RangeFinder](@ref). 
    With high probability we will have ``\\mathbb{E} \\|A - QQ^\\top A\\|_F \\leq
    \\sqrt{k+1} (\\sum_{i=k+1}^{\\min{(m,n)}}\\sigma_{i})^{1/2}``
    , where ``\\sigma_{k+1}`` is the ``k+1^\\text{th}`` singular value of (see Theorem 10.5 
    of [halko2011finding](@cite)). This bound is often conservative when the singular 
    values of ``A`` decay quickly. 

When the singular values decay slowly, we can apply ``A`` and ``A^\\top``, ``q`` times 
    and take the qr factorization of ``(AA^\\top)^q AS``, know as power iterations. 
    Using these power iterations increases the relative gap between the singular values, 
    which leads to  better RandomizedSVD performance. 

Power iterations can be unstable. Luckily, their stability  can be improved by 
    orthogonalizing ``AS`` after each application of ``A`` and ``A^\\top`` in what is known 
    as the orthogonalized power iteration. After computing ``Q`` the RandomizedSVD 
    concludes by computing ``W,S,V = \\text{SVD}(Q^\\top A)`` and  setting ``U = QW``.

# Fields
- `compressor::Compressor`, the technique for compressing the matrix from the right.
- `power_its::Int64`, the number of power iterations that should be performed.
- `orthogonalize::Bool`, a boolean indicating whether the `power_its` should be performed 
    with orthogonalization.
-`block_size::Int64`, the size of the tile when performing matrix multiplication. By 
    default, `block_size = 0`, this will be set to be the number of columns in 
    the original matrix.
"""
mutable struct RandSVD <: RangeApproximator
    compressor::Compressor
    power_its::Int64
    orthogonalize::Bool
    block_size::Int64
    function RandSVD(compressor, power_its, orthogonalize, block_size)
        if power_its < 0
            return throw(ArgumentError("Field `power_its` must be non-negative."))
        end

        if block_size < 0
            return throw(ArgumentError("Field `block_size` must be non-negative."))
        end
        
        return new(compressor, power_its, orthogonalize, block_size)
    end

end

RandSVD(;
    compressor = SparseSign(), 
    orthogonalize = false, 
    power_its = 1,
    block_size = 0,
) = RandSVD(compressor, power_its, orthogonalize, block_size)

"""
    RandSVDRecipe

A struct that contains the preallocated memory and completed compressor to form a
    RandSVD approximation to the matrix ``A``.

# Fields
- `n_rows::Int64`, the number of rows in the approximation. 
- `n_cols::Int64`, the number of columns in the approximation. 
- `compressor::CompressorRecipe`, the compressor to be applied from the right to ``A``.
- `power_its::Int64`, the number of power iterations that should be performed.
- `orthogonalize::Bool`, a boolean indicating whether the `power_its` should be performed 
    with orthogonalization.
- `U::AbstractArray`, the orthogonal matrix that approximates the top `compressor_dim` 
    left singular vectors of ``A``.
- `S::AbstractVector`, a vector containing the top `compressor_dim` singular values of 
    ``A``.
- `V::AbstractArray`, the orthogonal matrix that approximates the top `compressor_dim` 
    right singular vectors of ``A``.
-`buffer::AbstractArray`, the storage for matrix multiplication with this low-rank 
    approximation.
"""
mutable struct RandSVDRecipe <: RangeApproximatorRecipe
    n_rows::Int64
    n_cols::Int64
    compressor::CompressorRecipe
    power_its::Int64
    orthogonalize::Bool
    U::AbstractArray
    S::AbstractVector
    V::AbstractArray
    buffer::AbstractArray
end

function complete_approximator(approx::RandSVD, A::AbstractMatrix)
    type = eltype(A)
    # You need to make sure you orient the compressor in the correct direction
    if typeof(approx.compressor.cardinality) <: Left
        @warn "Compressor with cardinality `Left` being applied from `Right`."
    end

    compress = complete_compressor(approx.compressor, A)
    # Determine the dimensions of the range approximator
    a_rows, a_cols = size(A)
    c_cols = size(compress, 2)
    # if the blocksize zero set to be number of columns in A
    bsize = approx.block_size == 0 ? size(A, 2) : approx.block_size
    approx_recipe = RandSVDRecipe(
        a_rows,
        a_cols,
        compress, 
        approx.power_its,
        approx.orthogonalize, 
        Matrix{type}(undef, 2, 2),
        Vector{type}(undef,2),
        Matrix{type}(undef, 2, 2),
        Matrix{type}(undef, c_cols, bsize) 
    )
end

function rapproximate!(approx::RandSVDRecipe, A::AbstractMatrix)
    # User may wish to choose to use a different power iteration
    
    if approx.orthogonalize 
        Q = rand_ortho_it(A, approx) 
    else
        Q = rand_power_it(A, approx)
    end

    QA = Matrix{Float64}(undef, size(Q, 2), size(A, 2))
    # Making Q an Array is far more efficient than not
    mul!(QA, Q', A)
    U, approx.S, approx.V = svd(QA)
    approx.U = Q * U
    return nothing
end

function rapproximate(approx::RandSVD, A::AbstractMatrix)
    approx_recipe = complete_approximator(approx, A)
    rapproximate!(approx_recipe, A)
    return  approx_recipe
end

# only need to implement left and right muls because of transpose defs in main file
function mul!(
    C::AbstractArray, 
    R::RandSVDRecipe, 
    A::AbstractArray, 
    alpha::Number, 
    beta::Number
)
    left_mul_dimcheck(C, R, A)
    buff_size = size(R.buffer, 2)
    a_cols = size(A, 2)
    n_its = div(a_cols, buff_size)
    remaining_vals = rem(a_cols, buff_size)
    start_idx = 1
    for i in 1:n_its
        last_idx = start_idx + buff_size - 1
        # define views at current blocks of A and C
        Av = view(A, :, start_idx:last_idx)
        Cv = view(C, :, start_idx:last_idx)
        # apply the V matrix to the current block of A, use zero to ensure buffer is zeroed
        mul!(R.buffer, R.V', Av, alpha, 0.0)
        # apply diagonal matrix
        R.buffer .*= R.S
        # apply U and store in current block of C indices and scale C with beta
        mul!(Cv, R.U, R.buffer, 1.0, beta);
        start_idx = last_idx + 1
    end

    if remaining_vals != 0
        last_idx = start_idx + remaining_vals - 1
        # define views at current blocks of A and C
        Av = view(A, :, start_idx:last_idx)
        Cv = view(C, :, start_idx:last_idx)
        # define view at only necessary portion of buffer
        buffer_v = view(R.buffer, :, 1:remaining_vals)
        # apply the V matrix to the current block of A, use zero to ensure buffer is zeroed
        mul!(buffer_v, R.V', Av, alpha, 0.0)
        # apply diagonal matrix
        buffer_v .*= R.S
        # apply U and store in current block of C indices and scale C with beta
        mul!(Cv, R.U, buffer_v, 1.0, beta);
        
    end
    
    return nothing

end


function mul!(
    C::AbstractArray, 
    A::AbstractArray, 
    R::RandSVDRecipe, 
    alpha::Number, 
    beta::Number
)
    right_mul_dimcheck(C, A, R)
    buff_size = size(R.buffer, 2)
    a_rows = size(A, 1)
    n_its = div(a_rows, buff_size)
    remaining_vals = rem(a_rows, buff_size)
    start_idx = 1
    for i in 1:n_its
        last_idx = start_idx + buff_size - 1
        # define views at current blocks of A and C
        Av = view(A, start_idx:last_idx, :)
        Cv = view(C, start_idx:last_idx, :)
        # apply the V matrix to the current block of A, use zero to ensure buffer is zeroed
        mul!(R.buffer', Av, R.U, alpha, 0.0)
        # apply diagonal matrix (DA')' = AD
        R.buffer .*= R.S
        # apply U and store in current block of C indices and scale C with beta
        mul!(Cv, R.buffer', R.V', 1.0, beta);
        start_idx = last_idx + 1
    end

    if remaining_vals != 0
        last_idx = start_idx + remaining_vals - 1
        # define views at current blocks of A and C
        Av = view(A, start_idx:last_idx, :)
        Cv = view(C, start_idx:last_idx, :)
        # define view at only necessary portion of buffer because buffer is allocated in 
        # fashion where the number of rows is k and the number of columns is block_size
        # we still want this view to be over the columns even though the views for A and C
        # are on the rows
        buffer_v = view(R.buffer, :, 1:remaining_vals)
        # apply the V matrix to the current block of A, use zero to ensure buffer is zeroed
        mul!(buffer_v', Av, R.U, alpha, 0.0)
        # apply diagonal matrix (DA')' = AD
        buffer_v .*= R.S
        # apply U and store in current block of C indices and scale C with beta
        mul!(Cv, buffer_v', R.V', 1.0, beta);
        
    end
    
    return nothing
end
