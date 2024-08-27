
"""
    LinSysBlockColSRHT <: LinSysBlkColSampler

A mutable structure with fields to handle SRHT column sketching. For this procedure,
the hadamard transform and random sign swaps are applied once, then that matrix is repeatably
sampled.

# Fields
- `blockSize::Int64`, the size of blocks being chosen
- `paddedSize::Int64`, the size of the matrix when padded
- `block::Union{Vector{Int64}, Nothing}`, storage for block indices
- `Ap::Union{AbstractMatrix, Nothing}`, storage for padded matrix
- `signs::Union{Vector{Bool}, Nothing}`, storage for random sign flips.
- `scaling::Float64`, storage for the scaling of the sketches.

Calling `LinSysBlockColSRHT()` defaults to setting `blockSize` to 2.

Nir Ailon and Bernard Chazelle. 2006. Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform. In Proceedings of the thirty-eighth annual ACM symposium on Theory of Computing (STOC '06). Association for Computing Machinery, New York, NY, USA, 557–563. https://doi.org/10.1145/1132516.1132597
"""
mutable struct LinSysBlockColSRHT <: LinSysBlkColSampler
    blockSize::Int64
    paddedSize::Int64
    block::Union{Vector{Int64}, Nothing}
    Ap::Union{AbstractMatrix, Nothing}
    signs::Union{Vector{Bool}, Nothing}
    scaling::Float64
end

LinSysBlockColSRHT(blockSize) = LinSysBlockColSRHT(
                                                   blockSize, 
                                                   0, 
                                                   nothing, 
                                                   nothing,
                                                   nothing,
                                                   0.0
                                                  )
LinSysBlockColSRHT() = LinSysBlockColSRHT(2, 0, nothing, nothing, nothing, 0.0)

# Common sample interface for linear systems
function sample(
    type::LinSysBlockColSRHT,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        m, n = size(A)
        # If matrix is not a power of 2 then pad the rows
        if rem(log(2, n), 1) != 0
            type.paddedSize = Int64(2^(div(log(2, n), 1) + 1)) 
            # Find nearest power 2 and allocate
            type.Ap = zeros(m, type.paddedSize)
            # Pad matrix and constant vector
            type.Ap[:, 1:n] .= A
        else
            type.paddedSize = n
            type.Ap = A
        end
        # Compute scaling and sign flips
        type.scaling = sqrt(type.blockSize / type.paddedSize)
        type.signs = bitrand(type.paddedSize)
        for i = 1:m
            @views fwht!(type.Ap[i, :], signs = type.signs, scaling = type.scaling)
        end
        
        type.block = zeros(Int64, type.blockSize) 
    end

    type.block .= randperm(type.paddedSize)[1:type.blockSize] 
    AS = type.Ap[:, type.block]
    # Residual of the linear system
    res = A * x - b
    grad = AS'res
    H = hadamard(type.paddedSize)
    sgn = [type.signs[i] ? 1 : -1 for i in 1:type.paddedSize]
    return [sgn .* type.scaling, type.block], AS, res, grad
end
