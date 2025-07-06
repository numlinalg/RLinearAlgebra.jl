"""
    dotu(a::AbstractVector, b::AbstractVector)

A function that computes the non conjugate dot product between two vectors. It is equivalent
    to calling `dot(conj(a), b)`.

# Arguments
- `a::AbstractArray`, a vector being dot producted (is labeled as a array to allow for 
    views).
- `b::AbstractArray`, a vector being dot producted (is labeled as a array to allow for 
    views).
# Returns
- A scalar that is the non-conjugated dot product between two vectors.
"""
function dotu(a::AbstractArray, b::AbstractArray)
    n_a = maximum(size(a))
    n_b = maximum(size(b))
    if n_a != n_b
        throw(DimensionMismatch("Vector `a` and Vector `b` must be the same size."))
    end

    accum = zero(eltype(a))
    @simd for i in 1:n_a
        @inbounds accum += a[i] * b[i]
    end

    return accum   

end
