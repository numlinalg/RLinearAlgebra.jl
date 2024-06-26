"""
    LinSysVecBlkColReplace <: LinSysBlkColSampler 

A mutable structure with fields to handle randomly subset block sampling with replacement, also known as
vanilla block randomized kaczmarz. 

# Fields
- `block_size::Int64` - Specifies the size of each block.
- `block::Vector{Int64}` - The list of all the rows in each block.

Calling `LinSysBlkColReplace()` defaults to setting `block_size` to 2. The `sample`
function will handle the re-initialization of the fields once the system is provided.
"""
mutable struct LinSysBlkColReplace <: LinSysBlkColSampler 
    block_size::Int64
    block::Vector{Int64}
end

LinSysBlkColReplace(block_size) = LinSysBlkColReplace(block_size, Int64[])
LinSysBlkColReplace() = LinSysBlkColReplace(2, Int64[])

# Common sample interface for linear systems
function sample(
    type::LinSysBlkColReplace,
    A::AbstractArray,
    b::AbstractVector,
    x::AbstractVector,
    iter::Int64
)
    n = size(A, 2)
    if iter == 1
        @assert type.block_size <= n "Block size must be less than col dimension"
        type.block = zeros(Int64, type.block_size)
    end
    
    #Sample the indices using the sample function from StatsBase
    type.block = randperm(n)[1:type.block_size] 
    AS = A[:, type.block]
    res = A * x - b
    # Normal equation residual in the Sketched Block
    grad = AS' * (A * x - b)
    S = zeros(n, type.block_size)
    [S[type.block[i], i] = 1 for i in 1:type.block_size]

    return S, AS, grad, res
end
