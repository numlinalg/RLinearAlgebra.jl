"""
    LinSysBlockRowFJLT <: LinSysBlkRowSampler

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

Calling `LinSysBlockRowFJLT()` defaults to setting `sparsity` to .3 and the blocksize to 2.

Nir Ailon and Bernard Chazelle. 2006. Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform. In Proceedings of the thirty-eighth annual ACM symposium on Theory of Computing (STOC '06). Association for Computing Machinery, New York, NY, USA, 557–563. https://doi.org/10.1145/1132516.1132597
"""
mutable struct LinSysBlockRowFJLT <: LinSysBlkRowSampler
    blockSize::Int64
    sparsity::Float64 
    paddedSize::Int64
    Sketch::Union{SparseMatrixCSC, Nothing}
    Ap::Union{AbstractMatrix, Nothing}
    bp::Union{AbstractVector, Nothing}
    signs::Union{Vector{Bool}, Nothing}
    scaling::Float64
end

LinSysBlockRowFJLT(;blocksize = 2, sparsity = .3) = LinSysBlockRowFJLT(
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
    type::LinSysBlockRowFJLT,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        m, n = size(A)
        type.paddedSize = m
        # If matrix is not a power of 2 then pad the rows
        if rem(log(2, m), 1) != 0
            type.paddedSize = Int64(2^(div(log(2, m),1) + 1)) 
            # Find nearest power 2 and allocate
            type.Ap = zeros(type.paddedSize, n)
            type.bp = zeros(type.paddedSize)
            # Pad matrix and constant vector
            type.Ap[1:m, :] .= A
            type.bp[1:m] .= b
        else
            type.Ap = A
            type.bp = b
        end
        # Compute scaling and sign flips
        type.scaling = sqrt(type.blockSize / (type.paddedSize * type.sparsity))
        type.signs = bitrand(type.paddedSize)
        # Apply FWHT to padded matrix and vector
        fwht!(type.bp, signs = type.signs, scaling = type.scaling)
        for i = 1:n
            @views fwht!(type.Ap[:, i], signs = type.signs, scaling = type.scaling)
        end
        
    end

    type.Sketch = sprandn(type.blockSize, type.paddedSize, type.sparsity) 
    SA = type.Sketch * type.Ap
    Sb = type.Sketch * type.bp
    # Residual of the linear system
    res = SA * x - Sb
    return type.Sketch, SA, res
end
