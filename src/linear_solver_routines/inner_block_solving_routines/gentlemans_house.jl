# This is file from RLinearAlgebra.jl
"""
    GentData{S, M<:Matrix{S}, V<:Vector{S}}

This is a preallocated data structure for solving least squares problems with 
Gentleman's algorithm. This data structure is not exported and thus is not 
designed to be manipulated by the user.
# Fields
- `A::M`, The matrix to be solved.
- `B::M`, The matrix that rows of `A` are brought to have QR decomposition
applied.
- `R::UpperTriangular`, The upper triangular part of `B`.
- `tab::SubArray`, the last column of `B` where the constant vector entries corresponding
to the rows of `A` that were brought to `B` are stored.
- `v::V`, a buffer vector.
- `bsize::Int64`, the number of rows transported to `B` at each iteration.

Calling `Gent(A, bsize)` will allocate a `B` matrix with the n + 1 columns and 
n + bsize + 1 rows where n is the number of columns of A. It also allocates a 
buffer vector `v` with n + 1 entries. Finally, it takes a view of the upper
triangular part of `B[1:n, 1:n]` and a view of the last column of `B`.

Miller, Alan J. "Algorithm AS 274: Least squares routines to supplement those of Gentleman." Applied Statistics (1992): 458-478.
"""
mutable struct GentData{S, M<:Matrix{S}, V<:Vector{S}}
    A::M 
    B::M
    R::UpperTriangular
    tab::SubArray
    v::V
    bsize::Int64
end

Base.eltype(::GentData{S, M, V}) where {S, M, V} = S

# Constructor for Gent data structure
function Gent(A::AbstractMatrix, bsize::Int64)
    S = eltype(A)
    m,n = size(A)
    B = zeros(S, n + bsize + 1, n + 1)
    R = @views UpperTriangular(B[1:n, 1:n])
    tab = @views B[1:n, n + 1]
    v = zeros(n + 1) #zeros(n + bsize + 1)
    return GentData{S, Matrix{S}, Vector{S}}(A, B, R, tab, v, bsize)
end

"""
    gentleman!(G::GentData)

A function that updates the `GenData` structure with the Gentleman's
algorithm applied to a single row block of `A`.

# Inputs
- `G::GentData`, A data structure with the information to be updated
by an iteration of Gentlemans.

# Output
The function will result in updates to the upper triangular part `R`
and the `tab` vector of the `GentData` structure.
"""
function gentleman!(G::GentData)
    B = G.B
    v = G.v
    m, n = size(B)
    # Perform qr decomposition on B
    LAPACK.geqrf!(B,v)
    # Zero out the everything in the lower triangular part
    for i = 1:n
        for j = (i+1):m
            @inbounds B[j,i] = 0
        end

    end

end

#Function that zeros the gentleman's memory
"""
    resetGent!(G::GentData)

Function that sets all the allocated memory of
`GentData` to zero.
"""
function resetGent!(G::GentData)
    S = eltype(G)
    fill!(G.B, zero(S))
    fill!(G.v, zero(S))
end

#Least squares solver for Gentleman's
"""
    ldiv!(x::AbstractVector, G::GentData, b::AbstractVector)

Function that takes the allocated `GentData` structure to solve
the least squares problem \$\\min_x \\|A x-b\\|_2^2\$. The function
overwrites `x` with the solution.
"""
function LinearAlgebra.ldiv!(x::AbstractVector, G::GentData, b::AbstractVector)
    m,n = size(G.A)
    bsize = G.bsize
    remb = rem(m, bsize)
    # Detemine how many blocks by dividing by bsize and adding one if there is a remainder
    nblocks = div(m, bsize) + (remb > 1 ? 1 : 0)
    brows = n + bsize + 1
    for i in 1:nblocks
        # Do not use an index greater than m
        index = (i - 1) * bsize + 1: min(i * bsize, m)
        copyBlockFromMat!(G.B, G.A, b, index)
        # Check if you are in the last block in which case zero all rows 
        # that no data was moved to
        if index[end] < i * bsize
            fill!(view(G.B, (length(index) + 1) + n + 2:brows, :), zero(eltype(A)))
        end

        gentleman!(G)
    end

    # Solve Upper triangular system 
    LinearAlgebra.ldiv!(x, G.R, G.tab)
    resetGent!(G)
    return nothing 

end

# Function that copies the new block to the matrix being orthogonalized
"""
    copyBlockFromMat!(B::AbstractMatrix, A::AbstractMatrix, b::AbstractVector, index::Union{UnitRange{Int64}, Vector{Int64}})

Function that updates matrix `B` with the entries at indicies `index` of the `A` matrix and `b` constant vector of the linear 
system.
"""
function copyBlockFromMat!(B::AbstractMatrix, A::AbstractMatrix, b::AbstractVector, index::Union{UnitRange{Int64}, Vector{Int64}})
    m, n = size(B)
    l = length(index)
    # The upper triangular part is n by n
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