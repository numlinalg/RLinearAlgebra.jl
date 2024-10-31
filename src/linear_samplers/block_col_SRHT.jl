
"""
    LinSysBlockColSRHT <: LinSysBlkColSampler

A mutable structure with fields to handle SRHT column sketching. For this procedure,
the hadamard transform and random sign swaps are applied once, then that matrix is repeatably
sampled.

# Fields
- `block_size::Int64`, the size of blocks being chosen
- `padded_size::Int64`, the size of the matrix when padded
- `block::Union{Vector{Int64}, Nothing}`, storage for block indices
- `hadamard::Union{AbstractMatrix, Nothing}`, storage for the hadamard matrix.
- `Ap::Union{AbstractMatrix, Nothing}`, storage for padded matrix
- `signs::Union{Vector{Bool}, Nothing}`, storage for random sign flips.
- `scaling::Float64`, storage for the scaling of the sketches.

Calling `LinSysBlockColSRHT()` defaults to setting `block_size` to 2.

Nir Ailon and Bernard Chazelle. 2006. Approximate nearest neighbors and the fast Johnson-Lindenstrauss transform. In Proceedings of the thirty-eighth annual ACM symposium on Theory of Computing (STOC '06). Association for Computing Machinery, New York, NY, USA, 557–563. https://doi.org/10.1145/1132516.1132597
"""
mutable struct LinSysBlockColSRHT <: LinSysBlkColSampler
    block_size::Int64
    padded_size::Int64
    block::Union{Vector{Int64}, Nothing}
    hadamard::Union{AbstractMatrix, Nothing}
    Ap::Union{AbstractMatrix, Nothing}
    signs::Union{Vector{Bool}, Nothing}
    scaling::Float64
end

LinSysBlockColSRHT(block_size) = LinSysBlockColSRHT(
                                                   block_size, 
                                                   0, 
                                                   nothing, 
                                                   nothing,
                                                   nothing,
                                                   nothing,
                                                   0.0
                                                  )
LinSysBlockColSRHT() = LinSysBlockColSRHT(2, 0, nothing, nothing, nothing, nothing, 0.0)

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
            type.padded_size = Int64(2^(div(log(2, n), 1) + 1)) 
            # Find nearest power 2 and allocate
            type.Ap = zeros(m, type.padded_size)
            # Pad matrix and constant vector
            type.Ap[:, 1:n] .= A
        else
            type.padded_size = n
            type.Ap = A
        end

        type.hadamard = hadamard(type.padded_size)
        # Compute scaling and sign flips
        type.scaling = sqrt(1 / type.block_size)
        type.block = zeros(Int64, type.block_size) 
    end

    sgn = rand([-1, 1], type.padded_size)
    type.block .= randperm(type.padded_size)[1:type.block_size] 
    AS = (type.scaling * (type.Ap * sgn .* type.hadamard))[:, type.block]
    # Residual of the linear system
    res = A * x - b
    grad = AS'res
    return (type.scaling * sgn .* type.Hadamard)[:, type.block], AS, res, grad
end
