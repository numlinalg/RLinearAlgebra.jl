# Design philosophy: we wat "deep modules" with simple interfaces but deep functionality. We
# can achieve this with multidispatch. We allow users to access implementations directly but
# we provide a simple, common interface to those implementations following the example of
# PETSc (KSP module for linear solvers).
using LinearAlgebra
using Krylov
using Random

include("blendenpik_gauss.jl")
include("matrix_sampler.jl")
include("projection.jl")


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

struct TypeBlendenpik <: LinearSolverType end

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
    type = sol.type
    solve(sol, type, A, b)
end

function solve(sol::LinearSolver, type::TypeBlendenpik, A, b)
    return blendenpick_gauss(A, b, verbose=sol.verbose)
end

function solve(sol::LinearSolver, type::TypeRPM, A, b)
    # Retrieve sampler and projection types
    sampler_type = type.sampler
    projection_type = type.projection

    #Stopping threshold
    x_init = copy(b)
    x_init .= 0.0

    residual = norm(A*x_init - b)
    thresh = residual*sol.atol
    maxit = sol.maxit

    x = x_init
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
