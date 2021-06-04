# Implement RPM method as preconditioner
using RLinearAlgebra
using Krylov
import LinearAlgebra: ldiv!, \, *, mul!
import Base: eltype

# PRECONDITIONER DEFINITION
abstract type AbstractPreconditioner end
struct RPMPreconditioner{T, S<:AbstractMatrix{T}} <: AbstractPreconditioner
    solver::LinearSolver
    A::Matrix
end

function RPMPreconditioner(solver::LinearSolver, A::Matrix)
    return RPMPreconditioner{eltype(A), typeof(A)}(solver, A)
end

@inline function (*)(C::RPMPreconditioner, b::AbstractVector)
    x = zero(b)
    rsolve!(x, C.solver, C.A, b)
    return x
end

eltype(::RPMPreconditioner{T, S}) where {T, S} = T

# System
A = [1.0 0.5; 0.5 2.0];
b = [1.0, 1.0];

# Create solver
sol = LinearSolver(TypeRPM(SamplerKaczmarzCYC()))
P = RPMPreconditioner(sol, A)

# Solve with BICGSTAB
itmax = 1000
atol = 1e-10
rtol = 1e-10

println("Unpreconditioned")
(x, stats) = bicgstab(A, b, itmax=itmax, history=true, atol=atol, rtol=rtol)
println("\tIterations: ", length(stats.residuals))
println("\tExit message: ", stats.status)

println("Preconditioned")
(x, stats) = bicgstab(A, b, M=P, itmax=itmax, history=true, atol=atol, rtol=rtol)
println("\tIterations: ", length(stats.residuals))
println("\tExit message: ", stats.status)
