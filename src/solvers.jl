using LinearAlgebra
using Krylov
using Random

include("blendenpik_gauss.jl")
include("rpm_sampler.jl")
include("rpm_projection.jl")


# Abstract parent class for solver type
"""
    LinearSolverType

Abstract supertype that specifies the type of linear solver being deployed.
"""
abstract type LinearSolverType end
# Solver type

"""
    TypeRPM <: LinearSolverType

A struct that specifies a combination of the `RPMSamplerType` and the `RPMProjectionType`.

# Constructors

The default constructor, `TypeRPM()`, has `SamplerKaczmarzWR()` sampler type and
`ProjectionStdCore()` as the projection type.

`TypeRPM(sampler::RPMSamplerType)` allows for the sampler to be specified and defaults the
projection type to `ProjectionStdCore()`.

`TypeRPM(projection::RPMProjectionType)` allows for the projection type to be specified and
defaults the sampler type to `SamplerKaczmarzWR()`.
"""
struct TypeRPM <: LinearSolverType
    sampler::RPMSamplerType
    projection::RPMProjectionType
end
TypeRPM() = TypeRPM(SamplerKaczmarzWR(), ProjectionStdCore()) # default values
TypeRPM(sampler::RPMSamplerType) = TypeRPM(sampler, ProjectionStdCore()) # default values
TypeRPM(projection::RPMProjectionType) = TypeRPM(SamplerKaczmarzWR(), projection) # default values

"""
    TypeRGS <: LinearSolverType

A structure for encapuslating randomized Gauss-Seidel. No fields.
"""
struct TypeRGS <: LinearSolverType end

"""
    TypeBlendenpik <: LinearSolverType

A structure for encapsulating Blendenpik methods. No fields.

See Avron, Haim, Petar Maymounkov, and Sivan Toledo. "Blendenpik: Supercharging LAPACK's
least-squares solver." SIAM Journal on Scientific Computing 32, no. 3 (2010): 1217-1236.
"""
struct TypeBlendenpik <: LinearSolverType end

# Loggers
"""
    SolveLog

A mutable structure for tracking and recording important solver information.

# Fields

- `iters::Int64`, records the number of iterations
- `residual_hist::Vector{Float64}`, records the residual norm history
- `converged::Bool`, indicates whether a convergence metric is achieved

Calling `SolverLog()` initializes the structure to `(0, Vector{Float64}(undef, 0), false)`
"""
mutable struct SolveLog
    iters::Int64
    residual_hist::Vector{Float64}
    converged::Bool
end
SolveLog() = SolveLog(0, Vector{Float64}(undef, 0), false)

# Solver data structure
"""
    LinearSolver

A structure for containing information about a particular linear solver attempt on a given
system.

# Fields

- `type::LinearSolverType`, encapsulates the parameters of the linear solvers
- `maxit::Int64`, maximum number of iterations allowed for the solver
- `rtol::Float64`, relative tolerance stop criteria for the residual
- `atol::Float64`, absolute tolerance stop criteria for the residual
- `verbose::Bool`, unused
- `log::SolveLog`, solver log
"""
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
"""
    rsolve(sol::LinearSolver, A, b)

A wrapper for solving a linear system using the specified linear solver on a system
specified by coefficient matrix `A` and constant vector `b`. Returns a solution vector
`x`.
"""
function rsolve(sol::LinearSolver, A, b)
    x = zeros(size(A, 2))
    rsolve!(x, sol, A, b)
    return x
end

"""
    rsolve!(x, sol::LinearSolver, A, b)

An in-place update for solving a linear system with coefficient matrix `A` and constant
vector `b`. The solution is updatd in `x`.
"""
function rsolve!(x, sol::LinearSolver, A, b)
    type = sol.type
    rsolve!(x, sol, type, A, b)
    return nothing
end

function rsolve!(x, sol::LinearSolver, type::TypeBlendenpik, A, b)
    blendenpick_gauss!(x, A, b, verbose=sol.verbose)
    return nothing
end

function rsolve!(x, sol::LinearSolver, type::TypeRPM, A, b)
    projection = type.projection
    rsolve!(x, sol, type, projection, A, b)

    return nothing
end

function rsolve!(
    x,
    sol::LinearSolver,
    type::TypeRPM,
    projection::RPMProjectionType,
    A,
    b
)
    # Retrieve sampler
    sampler_type = type.sampler
    projection_type = type.projection

    residual = norm(A*x - b)
    thresh = residual*sol.atol
    maxit = sol.maxit

    j = 1
    while (j < maxit) & (residual > thresh)
        q, s = sample(sampler_type, A, b, x, j)
        project!(projection_type, x, q[:, 1], s, j)
        residual = norm(A*x - b)
        j += 1
        push!(sol.log.residual_hist, residual)
    end

    sol.log.iters = j
    residual < sol.atol ? sol.log.converged = true : sol.log.converged = false

    return nothing
end

function rsolve!(x, sol::LinearSolver, type::TypeRGS, A, b)
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

    return nothing
end
