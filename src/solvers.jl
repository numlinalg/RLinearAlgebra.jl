using LinearAlgebra
using Krylov
using Random

include("blendenpik_gauss.jl")
include("rpm_sampler.jl")
include("rpm_projection.jl")


# Abstract parent class for solver type
abstract type LinearSolverType end
# Solver type
struct TypeRPM <: LinearSolverType
    sampler::RPMSamplerType
    projection::RPMProjectionType
end
TypeRPM() = TypeRPM(SamplerKaczmarzWR(), ProjectionStdCore()) # default values
TypeRPM(sampler::RPMSamplerType) = TypeRPM(sampler, ProjectionStdCore()) # default values
TypeRPM(projection::RPMProjectionType) = TypeRPM(SamplerKaczmarzWR(), projection) # default values

struct TypeRGS <: LinearSolverType end

struct TypeBlendenpik <: LinearSolverType end

# Loggers
mutable struct SolveLog
    iters::Int64
    residual_hist::Vector{Float64}
    converged::Bool
end
SolveLog() = SolveLog(0, Vector{Float64}(undef, 0), false)

# Solver data structure
mutable struct LinearSolver
    type::LinearSolverType
    maxit::Int64
    rtol::Float64
    atol::Float64
    verbose::Bool
    log::SolveLog
end
LinearSolver(type::LinearSolverType) = LinearSolver(type, 500, 1e-8, 1e-6, false, SolveLog())

# Solver APIs

function solve(sol::LinearSolver, A, b)
    x = zeros(size(A, 2))
    solve!(x, sol, A, b)
    return x
end

function solve!(x, sol::LinearSolver, A, b)
    type = sol.type
    solve!(x, sol, type, A, b)
end

function solve!(x, sol::LinearSolver, type::TypeBlendenpik, A, b)
    blendenpick_gauss!(x, A, b, verbose=sol.verbose)
end

function solve!(x, sol::LinearSolver, type::TypeRPM, A, b)
    # Retrieve sampler and projection types
    sampler_type = type.sampler
    projection_type = type.projection

    residual = norm(A*x - b)
    thresh = residual*sol.atol
    maxit = sol.maxit

    j = 1
    while (j < maxit) & (residual > thresh)
        q, s = sample(sampler_type, A, b, x, j)
        x = project(projection_type, x, q[:, 1], s)
        residual = norm(A*x - b)
        j += 1
        push!(sol.log.residual_hist, residual)
    end

    sol.log.iters = j
    residual < sol.atol ? sol.log.converged = true : sol.log.converged = false

    return x
end

function solve!(x, sol::LinearSolver, type::TypeRGS, A, b)
    residual = b - A*x
    residual_norm = norm(residual)
    thresh = residual_norm*sol.atol
    maxit = sol.maxit
    ncol = size(A, 2)

    j = 1
    while (j < maxit) & (residual_norm > thresh)
        col = rand(1:ncol)
        x[col] = x[col] + dot(A[:, col], residual)/dot(A[:, col], A[:, col])
        residual = b - A*x
        residual_norm = norm(residual)
        j += 1
        push!(sol.log.residual_hist, residual_norm)
    end
end
