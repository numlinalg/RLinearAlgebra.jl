"""
    LinSysBlkRowSRHT <: LinSysBlkRowSampler

A mutable structure with fields to handle SRHT row sketching. At each iteration, this 
procedure generates a matrix of the form S = R H D where R is a subset of the identity 
matrix, H is a Hadamard matrix, and D is a diagonal matrix with a rademacher vector on 
the diagonal.

# Fields
- `block_size::Int64`, the size of blocks being chosen
- `padded_size::Int64`, the size of the matrix when padded
- `block::Union{Vector{Int64}, Nothing}`, storage for block indices
- `hadamard::Union{AbstractMatrix, Nothing}`, storage for the hadamard matrix.
- `Ap::Union{AbstractMatrix, Nothing}`, storage for padded matrix
- `bp::Union{AbstractMatrix, Nothing}`, storage for padded vector
- `scaling::Float64`, storage for the scaling of the sketches.

Calling `LinSysBlkRowSRHT()` defaults to setting `block_size` to 2.

Nguyen, Nam H., Thong T. Do, and Trac D. Tran. "A fast and efficient algorithm for low-rank approximation of a matrix." 
Proceedings of the forty-first annual ACM symposium on Theory of computing. 2009. https://doi.org/10.1145/1536414.1536446
"""
mutable struct LinSysBlkRowSRHT <: LinSysBlkRowSampler
    block_size::Int64
    padded_size::Int64
    block::Union{Vector{Int64}, Nothing}
    hadamard::Union{AbstractMatrix, Nothing}
    Ap::Union{AbstractMatrix, Nothing}
    bp::Union{AbstractVector, Nothing}
    scaling::Float64
end

LinSysBlkRowSRHT(;block_size = 2) = LinSysBlkRowSRHT(
                                                   block_size, 
                                                   0, 
                                                   nothing, 
                                                   nothing,
                                                   nothing,
                                                   nothing,
                                                   0.0
                                                  )

# Common sample interface for linear systems
function sample(
    type::LinSysBlkRowSRHT,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    if iter == 1
        m, n = size(A)
        # If matrix is not a power of 2 then pad the rows
        if rem(log(2, m), 1) != 0
            type.padded_size = Int64(2^(div(log(2, m), 1) + 1)) 
            # Find nearest power 2 and allocate
            type.Ap = zeros(type.padded_size, n)
            type.bp = zeros(type.padded_size)
            # Pad matrix and constant vector
            type.Ap[1:m, :] .= A
            type.bp[1:m] .= b
        else
            type.padded_size = m
            type.Ap = A
            type.bp = b
        end

        type.hadamard = hadamard(type.padded_size)
        # Compute scaling and sign flips
        type.scaling = sqrt(1 / type.block_size)
        # Allocate a vector to contain a random permutation of the blocks
        type.block = zeros(Int64, type.block_size) 
    end

    sgn = rand([-1, 1], type.padded_size)
    type.block .= randperm(type.padded_size)[1:type.block_size] 
    SA = type.scaling * (type.hadamard * (sgn .* type.Ap))[type.block, :]
    Sb = type.scaling * (type.hadamard * (sgn .* type.bp))[type.block]
    # Residual of the linear system
    res = SA * x - Sb
    return ((sgn .* type.hadamard')' .* type.scaling)[type.block, :], SA, res
end