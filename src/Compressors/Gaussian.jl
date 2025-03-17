struct Gaussian <: Compressor
    n_rows::Int64
    n_cols::Int64
end

function Gaussian(;n_rows::Int64 = 0, n_cols::Int64 = 0)
    # Partially construct the Gaussian datatype
    return Gaussian(n_rows, n_cols)
end

mutable struct GaussianRecipe <: CompressorRecipe
    n_rows::Int64
    n_cols::Int64
    scale::Float64
    sketch_matrix::Matrix{Float64}
end

function complete_compressor(Gaussian_info::Gaussian, A::AbstractMatrix)
    n_rows = Gaussian_info.n_rows
    n_cols = Gaussian_info.n_cols
    # Find the zero dimension and set it to be the dimension of A
    if n_rows == 0 && n_cols == 0
        # By default we will compress the row dimension to size 2
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
            sample_size = n_rows
        else
            @assert false "Either you inputted row or column dimension must match \\
            the column or row dimension of the matrix."
        end

    end

    # Generate entry values by N(0,1/d)
    scale = 1 / sqrt(sample_size)
    sketch_matrix = scale .* randn(n_rows, n_cols) 
    return GaussianRecipe(n_rows, n_cols, scale, sketch_matrix)
end

# Allocations in this function are entirely due to bitrand call
function update_compressor!(S::GaussianRecipe)
    # Inplace update of sketch_matrix
    randn!(S.sketch_matrix)
    lmul!(S.scale, S.sketch_matrix)
    return
end

# Implement the matrix-vector multiplication
# Do the right version
function mul!(x::AbstractVector, S::GaussianRecipe, y::AbstractVector, alpha::Real, beta::Real)
    # Check the compatability of the sizes of the things being multiplied
    vec_mul_dimcheck(x, S, y)
    mul!(x, S.sketch_matrix, y, alpha, beta)
    return
end

function mul!(x::AbstractVector, S::CompressorAdjoint{GaussianRecipe}, y::AbstractVector, alpha::Real, beta::Real)
    # Check the compatability of the sizes of the things being multiplied
    vec_mul_dimcheck(x, S, y)
    mul!(x, S.parent.sketch_matrix', y, alpha, beta)
    return
end

# Implement the matrix-Matrix Multiplication operators
# Begin with the left version
function mul!(C::AbstractMatrix, S::GaussianRecipe, A::AbstractMatrix, alpha::Real, beta::Real)
    left_mat_mul_dimcheck(C, S, A) 
    # Built-in multiplication
    mul!(C, S.sketch_matrix, A, alpha, beta)
    return
end

# Now implement the right versions
function mul!(C::AbstractMatrix, A::AbstractMatrix, S::GaussianRecipe, alpha::Real, beta::Real)
    right_mat_mul_dimcheck(C, A, S)
    # Built-in multiplication
    mul!(C, A, S.sketch_matrix, alpha, beta)
    return
end