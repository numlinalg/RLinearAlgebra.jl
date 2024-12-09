# Date: 11/27/2024
# Author: Christian Varner
# Purpose: Implement the modified gram schmidt method

"""
    mgs!(q::AbstractVector, basis::AbstractMatrix)

Perform the modified gram-schmidt to orthogonalize `q`
with respect to the set of vectors in `basis`. 

!!! note
    Edits the vector `q` in place.

# Arguments

- `q::AbstractVector`, vector that will be orthogonalized.
- `basis::AbstractMatrix`, basis to orthogonalize against.

# Returns

- `h::AbstractVector`, orthogonalizing coefficients.

"""
function mgs!(q::AbstractVector, basis::AbstractMatrix)
    
    # initializations
    sz = size(basis, 2)
    h = zeros(sz)
    
    # orthogonalization loop
    for i in 1:sz
        bi = view(basis, :, i)
        h[i] = dot(q, bi)
        q .-= h[i] .* bi
    end
    
    return h
end

"""
    mgs!(q::AbstractVector, h::AbstractVector, basis::AbstractMatrix)

Perform the modified gram-schmidt to orthogonalize `q` against
the vectors in `basis`.

!!! note
    Edits `q` in place, and stores the coefficients for orthogonalization
    in the vector h, which is assumed to be a view of another matrix.

# Arguments

- `q::AbstractVector`, vector that will be orthogonalized.
- `h::AbstractVector`, view of a matrix to store the orthogonalizing coefficients
- `basis::AbstractMatrix`, basis to orthogonalize against.
"""
function mgs!(q::AbstractVector, h::AbstractVector, basis::AbstractMatrix)

    sz = size(basis, 2)
    
    # orthogonalization loop
    for i in 1:sz
        bi = view(basis, :, i)
        h[i] = dot(q, bi)
        q .-= h[i] .* bi
    end
end




