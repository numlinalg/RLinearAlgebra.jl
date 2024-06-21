"""
    LinSysBlockColFJLT <: LinSysBlkColSampler

A mutable structure with fields to handle FJLT row sketching. For this procedure,
the hadamard transform and random sign swaps are applied once, then that matrix is repeatably
sampled.

# Fields
- `blockSize::Int64`, the size of the sketching dimension
- `sparsity::Float64`, the sparsity of the sampling matrix
- `paddedSize::Int64`, the size of the matrix when padded
- `Sketch::Union{SparseMatrixCSC, Nothing}`, storage for sparse sketching matrix 
- `Ap::Union{AbstractMatrix, Nothing}`, storage for padded matrix
- `bp::Union{AbstractMatrix, Nothing}`, storage for padded vector
- `signs::Union{Vector{Bool}, Nothing}`, storage for random sign flips.
- `scaling::Float64`, storage for the scaling of the sketches.

Calling `LinSysBlockColFJLT()` defaults to setting `sparsity` to .3 and the blocksize to 2.

Nir Ailon and Bernard Chazelle. 2006. Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform. In Proceedings of the thirty-eighth annual ACM symposium on Theory of Computing (STOC '06). Association for Computing Machinery, New York, NY, USA, 557–563. https://doi.org/10.1145/1132516.1132597
"""
mutable struct LinSysBlockColFJLT <: LinSysBlkColSampler
    blockSize::Int64
    sparsity::Float64 
    paddedSize::Int64
    Sketch::Union{SparseMatrixCSC, Nothing}
    Ap::Union{AbstractMatrix, Nothing}
    bp::Union{AbstractVector, Nothing}
    signs::Union{Vector{Bool}, Nothing}
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
        type.paddedSize = n
        # If matrix is not a power of 2 then pad the rows
        if rem(log(2, n), 1) != 0
            type.paddedSize = Int64(2^(div(log(2, n),1) + 1)) 
            # Find nearest power 2 and allocate
            type.Ap = zeros(m ,type.paddedSize)
            # Pad matrix and constant vector
            type.Ap[:, 1:n] .= A
        else
            type.Ap = A
        end
        # Compute scaling and sign flips
        type.scaling = sqrt(type.blockSize / (type.paddedSize * type.sparsity))
        type.signs = bitrand(type.paddedSize)
        # Apply FWHT to padded matrix and vector
        for i = 1:m
            @views fwht!(type.Ap[i, :], signs = type.signs, scaling = type.scaling)
        end
        
    end

    type.Sketch = sprandn(type.paddedSize, type.blockSize, type.sparsity) 
    AS = type.Ap * type.Sketch
    # Residual of the linear system
    res = A * x - b
    grad = AS'res
    H = hadamard(type.paddedSize)
    sgn = [type.signs[i] ? 1 : -1 for i in 1:type.paddedSize]
    return (Diagonal(sgn) * H) * type.Sketch .* type.scaling, AS, res, grad
end
