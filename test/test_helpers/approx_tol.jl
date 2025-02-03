module ApproxTol
    # Modify ≈ to have a global tolerance
    ATOL = 1e-10
    import Base.≈
    ≈(a::Float64, b::Float64) = isapprox(a, b, atol = ATOL)
    # Adjust for tor the sum of multiple elements
    ≈(a::AbstractArray, b::AbstractArray) = isapprox(a, b, atol = .5 * ATOL * prod(size(a)))
    export ≈
end

