# Design philosophy: we wat "deep modules" with simple interfaces but deep functionality. We
# can achieve this with multidispatch. We allow users to access implementations directly but
# we provide a simple, common interface to those implementations following the example of
# PETSc (KSP module for linear solvers).
module Solvers

using LinearAlgebra
using Krylov
using Random

include("blendenpik_gauss.jl")
include("matrix_sampler.jl")
include("projection.jl")

# Abstract parent class for solver type
abstract type LinearSolverType end
# Solver type
struct TypeRPM <: LinearSolverType end
struct TypeBlendenpik <: LinearSolverType end

# Solver data structure
abstract type LinearSolverStruct end
mutable struct LinearSolver <: LinearSolverStruct
    type::LinearSolverType
    maxit::Int64
    rtol::Float64
    atol::Float64
    verbose::Bool
end
LinearSolver(type::LinearSolverType) = LinearSolver(type, 100, 1e-8, 1e-6, false)

# Solver APIs
function solve(sol::LinearSolverStruct, A, b)
    type = sol.type
    solve(sol, type, A, b)
end

function solve(sol::LinearSolverStruct, type::TypeBlendenpik, A, b)
    return blendenpick_gauss(A, b, verbose=sol.verbose)
end

function solve(sol::LinearSolverStruct, type::TypeRPM, A, b)
    #Stopping threshold
    x_init = copy(b)
    x_init .= 0.0

    thresh = norm(A*x_init - b)*sol.atol
    maxit = sol.maxit

    #Sampling
    sampler = kaczmarzWR(A, b)

    x = x_init
    j = 1
    while (j < maxit) & (norm(A*x - b) > thresh)
        q, s = sampler()
        x = stdCore(x, q[:, 1], s)
        j += 1
    end

    return x
end

# end of module
end
