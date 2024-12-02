# Date: 11/27/2024
# Author: Christian Varner
# Purpose: Implement the modified gram schmidt method

"""
    mgs!(q::AbstractVector, basis::Vector{Vector})

Perform the modified gram-schmidt to orthogonalize `q`
with respect to the set of vectors in `basis`. 

!!! Remark
    Edits the vector `q` in place.

# Arguments

- `q::AbstractVector`, vector that will be orthogonalized.
- `basis::AbstractMatrix`, basis to orthogonalize against.

# Returns

- `h::AbstractVector`, orthogonalizing coefficients.

"""
function mgs!(q::AbstractVector, basis::AbstractMatrix)
    
    # error checking
    @assert size(basis, 1) == size(q, 1) 
    "Vector `q` has length $(size(q, 1)) but should be $(size(basis, 1))."
    
    # initializations
    sz = size(basis, 2)
    @assert sz > 0 "No basis vectors to orthogonalize against."

    # orthogonalization loop
    h = zeros(sz)
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

!!! Remark
    Edits `q` in place, and stores the coefficients for orthogonalization
    in the vector h, which is assumed to be a view of another matrix.

# Arguments

- `q::AbstractVector`, vector that will be orthogonalized.
- `h::AbstractVector`, view of a matrix to store the orthogonalizing coefficients
- `basis::AbstractMatrix`, basis to orthogonalize against.
"""
function mgs!(q::AbstractVector, h::AbstractVector, basis::AbstractMatrix)


    # error checking
    @assert size(basis, 1) == size(q, 1) 
    "Vector `q` has length $(size(q, 1)), which is not equal to $(size(basis, 1))
    the number of rows in `basis`."

    sz = size(basis, 2)
    @assert sz > 0 "No basis vectors to orthogonalize against."
    @assert size(h, 1) == sz 
    "Size of `h` is $(size(h, 1)), which is not equal to $(sz) 
    the number of columns in `basis`."
    
    # orthogonalization loop
    for i in 1:sz
        bi = view(basis, :, i)
        h[i] = dot(q, bi)
        q .-= h[i] .* bi
    end
end




