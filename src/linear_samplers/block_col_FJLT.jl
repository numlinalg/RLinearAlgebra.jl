"""
    LinSysBlockColFJLT <: LinSysBlkColSampler

A mutable structure with fields to handle FJLT column sketching. For this procedure,
the sketching matrix S = D * H * G, where D is a diagonal matrix with a Rademacher vector
on the diagonal, H is a hadamard matrix, and G is a sparse Gaussian matrix. D is generated once
while G is generated everytime the `sample` function is called.

# Fields
- `block_size::Int64`, the size of the sketching dimension
- `sparsity::Float64`, the sparsity of the sampling matrix, should be between 0 and 1
- `padded_size::Int64`, the size of the matrix when padded
- `sampling_matrix::Union{SparseMatrixCSC, Nothing}`, storage for sparse sketching matrix 
- `hadamard::Union{AbstractMatrix, Nothing}`, storage for the hadamard matrix.
- `Ap::Union{AbstractMatrix, Nothing}`, storage for padded matrix
- `bp::Union{AbstractMatrix, Nothing}`, storage for padded vector
- `scaling::Float64`, storage for the scaling of the sketches.

Calling `LinSysBlockColFJLT()` defaults to setting `sparsity` to .3 and the blocksize to 2.

Nir Ailon and Bernard Chazelle. 2006. Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform. In Proceedings of the thirty-eighth annual ACM symposium on Theory of Computing (STOC '06). Association for Computing Machinery, New York, NY, USA, 557–563. https://doi.org/10.1145/1132516.1132597
"""
mutable struct LinSysBlockColFJLT <: LinSysBlkColSampler
    block_size::Int64
    sparsity::Float64 
    padded_size::Int64
    sampling_matrix::Union{SparseMatrixCSC, Nothing}
    hadamard::Union{AbstractMatrix, Nothing}
    Ap::Union{AbstractMatrix, Nothing}
    bp::Union{AbstractVector, Nothing}
    scaling::Float64
end

LinSysBlockColFJLT(;blocksize = 2, sparsity = .3) = LinSysBlockColFJLT(
                                                   blocksize,
                                                   sparsity, 
                                                   0, 
                                                   nothing, 
                                                   nothing,
                                                   nothing,
                                                   nothing,
                                                   0.0
                                                  )

# Common sample interface for linear systems
function sample(
    type::LinSysBlockColFJLT,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        m, n = size(A)
        # If matrix is not a power of 2 then pad the rows
        if rem(log(2, n), 1) != 0
            type.padded_size = Int64(2^(div(log(2, n), 1) + 1)) 
            # Find nearest power 2 and allocate
            type.Ap = zeros(m, type.padded_size)
            # Pad matrix and constant vector
            @views type.Ap[:, 1:n] .= A
        else
            type.padded_size = n
            type.Ap = A
        end
        type.hadamard = hadamard(type.padded_size)
        # Compute scaling and sign flips
        type.scaling = sqrt(type.block_size / (type.padded_size * type.sparsity))
        
    end

    sgn = rand([-1, 1], type.padded_size)
    type.sampling_matrix = sprandn(type.padded_size, type.block_size, type.sparsity) 
    AS = (type.Ap * (sgn .* type.hadamard)) * type.sampling_matrix * type.scaling
    # Residual of the linear system
    res = A * x - b
    grad = AS' * res
    return ((sgn .* type.hadamard) * type.sampling_matrix * type.scaling), AS, res, grad
end
