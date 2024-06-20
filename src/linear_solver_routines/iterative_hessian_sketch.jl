"""
"""

mutable struct IterativeHessianSketch <: LinSysSolveRoutine 
    A::AbstractArray
    b::AbstractVector
    step::Union{AbstractVector, Nothing}
    btilde::Union{AbstractVector, Nothing} 
end

function rsubsolve!(
    type::IterativeHessianSketch,
    x::AbstractVector,
    samp::Tuple{U,V,W} where {U<:AbstractArray,V<:AbstractArray,W<:AbstractVector},
    iter::Int64
)
    # samp[1] is the search direction
    # samp[2] is the coefficient matrix
    # samp[3] is the residual of the sketched system

    if iter == 1
        p = size(x)[1]
        type.step = Array{typeof(samp[2][1])}(undef, p) 
        type.btilde = Array{typeof(samp[2][1])}(undef, size(type.A)[2]) 
    end

    m = size(samp[1])[1]    
    type.btilde .= m .* ((type.A)'*(type.b - type.A*x))
    LinearAlgebra.ldiv!(type.step,lq(samp[2]'*samp[2]),type.btilde) # what does this do
    x .= x .+ type.step
end