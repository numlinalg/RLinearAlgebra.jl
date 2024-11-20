"""
    ApproxErrorMethod 

Abstract supertype for error methods for low rank approximations to matrices. 

"""
abstract type ApproxErrorMethod end

"""
    RangeFinderError <: ApproxErrorMethod 

Abstract supertype for error methods for Random Rangefinder techniques such as the random svd decomposition random qr decomposition, and the random eigen decompositions.
"""
abstract type RangeFinderError <: ApproxErrorMethod end

"""
    error_approximate!(error:T where T<: ApproxErrorMethod, LinApprox::ApproxMethod, A::AbstractMatrix)

Function that computes the error of an low rank approximation using a ApproxErrorMethod.
"""
function error_approximate!(error::T where T<: ApproxErrorMethod, LinApprox::ApproxMethod, A::AbstractMatrix)
    return nothing
end
