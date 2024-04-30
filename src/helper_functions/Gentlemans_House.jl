mutable struct GentData{S, M<:Matrix{S}, V<:Vector{S}}
    A::M 
    B::M
    R::UpperTriangular
    tab::SubArray
    v::V
    bsize::Int64
end

Base.eltype(::GentData{S, M, V}) where {S, M, V} = S

function Gent(A::AbstractMatrix, bsize::Int64)
    S = eltype(A)
    m,n = size(A)
    B = zeros(S, n + bsize + 1, n + 1)
    R = @views UpperTriangular(B[1:n, 1:n])
    tab = @views B[1:n, n + 1]
    v = zeros(n + 1) #zeros(n + bsize + 1)
    return GentData{S, Matrix{S}, Vector{S}}(A, B, R, tab, v, bsize)
end
#Apply Genlemans to the block
function gentleman!(G::GentData)
    B = G.B
    v = G.v
    m, n = size(B)
    LAPACK.geqrf!(B,v)
    for i = 1:n
        for j = (i+1):m
            @inbounds B[j,i] = 0
        end
    end
end
#Function that zeros the gentleman's memory
function resetGent(G::GentData)
    S = eltype(G)
    fill!(G.B, zero(S))
    fill!(G.v, zero(S))
end
#Least squares solver for Gentleman's
function ldiv!(x::AbstractVector, G::GentData, b::AbstractVector)
    m,n = size(G.A)
    bsize = G.bsize
    remb = rem(m, bsize)
    nblocks = div(m, bsize) + (remb > 1 ? 1 : 0)
    brows = n + bsize + 1
    for i in 1:nblocks
        index = (i - 1) * bsize + 1: min(i * bsize, m)
        copyBlockFromMat!(G.B, G.A, b, index)
        if index[end] < i * bsize
            fill!(view(G.B, (length(index) + 1)+n+2:brows, :), zero(eltype(A)))
        end
        gentleman!(G)
    end
    LinearAlgebra.ldiv!(x, G.R, G.tab)
    resetGent(G)
    return nothing 
end
# Function that copies the new block to the matrix being orthogonalized
function copyBlockFromMat!(B::AbstractMatrix, A::AbstractMatrix, b::AbstractVector, index)
    m, n = size(B)
    l = length(index)
    offset = n
    for i in 1:n-1
        for j in 1:l
            B[j + offset, i] = A[index[j], i]
        end
    end
    for j in 1:l
        B[j + offset, n] = b[index[j]]
    end
end
        
