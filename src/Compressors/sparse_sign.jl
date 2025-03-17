"""
   SparseSign <: Compressor

An implementation of the sparse sign sketching method. This method forms a sparse matrix 
    with a fixed number of non-zeros per row or column depending on the direction that the 
    sketch is being applied. If sketching from the left, then the number of nonzeros is 
    fixed for each column. For sketching from the right, then the number of nonzeros is
    fixed for each row. Each nonzero entry takes a value of 
    ``{-\\sqrt{\\text{nnz}}, \\sqrt{\\text{nnz}}}``  [martinsson2020randomized](@cite).

# Fields
 - `n_rows::Int64`, the number of rows in the sketching matrix.
 - `n_cols::Int64`, the number of columns in the sketching matrix.
 - `nnz::Int64`, the number of non-zero entries in each row if column compression or the 
 number of non-zero entries each column if row compression.
 - `scale::Float64`, the value of the non-zero entries.
 - `idxs::Vector{Int64}`, the index of each non-zero entry in each row if column 
 compression or each column if row compression. 
 - `signs::Vector{Bool}`, the sign of each non-zero entry in each row if column 
 compression or each column if row compression.  As a boolean vector `0`s indicate negative 
 signs while `1`s indicate positive signs.

# Constructor
Will be constructed with `SparseSign(n_rows, n_cols; nnz = 8)` where
- `n_rows::Int64`, the number of rows in the compression matrix. If compressing rows,
you should set it to be less than number of rows in the matrix you wish to compress. 
If you are compressing columns, you should set it to be same dimension as the number 
of columns in the matrix you wish to compress.
- `n_cols::Int64`, the number of columns in the compression matrix. If compressing columns,
you should set it to be less than number of columns in the matrix you wish to compress. 
If you are compressing rows, you should set it to be same dimension as the number 
of rows in the matrix you wish to compress.
- `nnz::Int64`, the number of non-zeros per row/column in the sampling matrix. By default 
this is set to 8.

"""
struct SparseSign <: Compressor
    n_rows::Int64
    n_cols::Int64
    nnz::Int64
end

function SparseSign(;n_rows::Int64 = 0, n_cols::Int64 = 0, nnz::Int64 = 8)
    # Partially construct the sparse sign datatype
    return SparseSign(n_rows, n_cols, nnz)
end

"""
    SparseSignRecipe <: CompressorRecipe

The recipe containing all allocations and information for the SparseSign compressor.

# Fields
 - `n_rows::Int64`, the number of rows in the sketching matrix.
 - `n_cols::Int64`, the number of columns in the sketching matrix.
 - `max_idx::Int64`, the maximum index in idxs.
 - `nnz::Int64`, the number of non-zero entries in each row if column compression or the 
 number of non-zero entries each column if row compression.
 - `scale::Float64`, the value of the non-zero entries.
 - `idxs::Vector{Int64}`, the index of each non-zero entry in each row if column 
 compression or each column if row compression. 
 - `signs::Vector{Bool}`, the sign of each non-zero entry in each row if column 
 compression or each column if row compression.  As a boolean vector `0`s indicate negative 
 signs while `1`s indicate positive signs.

"""
mutable struct SparseSignRecipe <: CompressorRecipe
    n_rows::Int64
    n_cols::Int64
    nnz::Int64
    scale::Float64
    Mat::SparseMatrixCSC
end

# This runs sparse sign assuming the keyword version had been previously run
function complete_compressor(sparse_info::SparseSign, A::AbstractMatrix)
    n_rows = sparse_info.n_rows
    n_cols = sparse_info.n_cols
    # get the element type of the matrix
    T = eltype(A)
    # FInd the zero dimension and set it to be the dimension of A
    if n_rows == 0 && n_cols == 0
        # by default we will compress the row dimension to size 2
        n_cols = size(A, 1)
        n_rows = 2
        # correct these sizes
        initial_size = max(n_rows, n_cols)
        sample_size = min(n_rows, n_cols)
    elseif n_rows == 0 && n_cols > 0
        # Assuming that if n_rows is not specified we compress column dimension
         n_rows = size(A, 2)
         # If the user specifies one size as nonzero that is the sample size
         sample_size = n_cols
         initial_size = n_rows
    elseif n_rows > 0 && n_cols == 0
        n_cols = size(A, 1)
        sample_size = n_rows
        initial_size = n_cols
    else
        if n_rows == size(A, 2)
            initial_size = n_rows
            sample_size = n_cols
        elseif n_cols == size(A, 2)
            initial_size = n_cols
            sample_size == n_rows
        else
            @assert false "Either you inputted row or column dimension must match \\
            the column or row dimension of the matrix."
        end
    end

    nnz = (sparse_info.nnz == 8) ? min(8, sample_size) : sparse_info.nnz
    @assert nnz <= sample_size "Number of non-zero indices, $nnz, must be less than compression
    dimension, $sample_size."
    idxs = Vector{Int64}(undef, nnz * initial_size)
    start = 1
    for i in 1:initial_size
        # every grouping of nnz entries corresponds to each row/column in sample
        stop = start + nnz - 1
        # Sample indices from the intial_size
        @views sample!(
            1:sample_size, 
            idxs[start:stop], 
            replace = false, 
            ordered = true
        )
        start = stop + 1
    end
    
    # Store signs as a boolean to save memory
    scale = 1 / sqrt(nnz)
    signs = rand([-scale, scale], nnz * initial_size)  
    ptr = collect(1:nnz:(nnz * initial_size + 1)) 
    #If left sketching store as a sparse matrix if right store as sparse matrix transpose
    if initial_size == n_rows
        Mat = SparseMatrixCSC{T, Int64}(n_cols, n_rows, ptr, idxs, signs)'
    else
        Mat = SparseMatrixCSC{T, Int64}(n_rows, n_cols, ptr, idxs, signs)
    end

    return SparseSignRecipe(n_rows, n_cols, nnz, scale, Mat)
end

# This runs sparse Sign with both a matrix and vector input (for linear solver)
function complete_compressor(sparse_info::SparseSign, A::AbstractMatrix, b::AbstractVector)
    complete_compressor(sparse_info, A)
end

# allocations in this function are entirely due to bitrand call
function update_compressor!(S::SparseSignRecipe)
    # Sample_size will be the minimum of the two size dimensions of `S`
    sample_size = min(S.n_rows, S.n_cols)
    initial_size = max(S.n_rows, S.n_cols)
    start = 1
    for i in 1:sample_size
        # every grouping of nnz entries corresponds to each row/column in sample
        stop = start + S.nnz - 1
        # Sample indices from the intial_size
        @allocated mv = view(S.Mat.rowval, start:stop)
        @allocated @views sample!(
            1:sample_size, 
            mv, 
            replace = false, 
            ordered = true
        )
        start = stop + 1
    end

    # There is no inplace update of bitrand and using sample is slower
    @allocated rand!(S.Mat.nzval, [S.scale, -S.scale])
    return 
end


# Do the right version
function mul!(
    x::AbstractVector, 
    S::SparseSignRecipe, 
    y::AbstractVector, 
    alpha::Float64, 
    beta::Float64
)
    # Check the compatability of the sizes of the things being multiplied
    vec_mul_dimcheck(x, S, y)
    mul!(x, S.Mat, y, alpha, beta)
    return
end

function mul!(
    x::AbstractVector, 
    S::CompressorAdjoint{SparseSignRecipe}, 
    y::AbstractVector, 
    alpha::Float64, 
    beta::Float64
)
    # Check the compatability of the sizes of the things being multiplied
    vec_mul_dimcheck(x, S, y)
    mul!(x, S.parent.Mat', y, alpha, beta)
    return
end

# Implement the matrix-Matrix Multiplication operators
# Begin with the left version
function mul!(
    C::AbstractMatrix, 
    S::SparseSignRecipe, 
    A::AbstractMatrix, 
    alpha::Float64, 
    beta::Float64
)
    left_mat_mul_dimcheck(C, S, A)
    mul!(C, S.Mat, A, alpha, beta)
end

# Now implement the right versions
function mul!(
    C::AbstractMatrix, 
    A::AbstractMatrix, 
    S::SparseSignRecipe, 
    alpha::Float64, 
    beta::Float64
)
    right_mat_mul_dimcheck(C, A, S)
    mul!(C, A, S.Mat, alpha, beta) 
    return
end

