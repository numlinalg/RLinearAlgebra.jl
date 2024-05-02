#Example for using the block samplers for rows and columns
using RLinearAlgebra
using LinearAlgebra
using UnicodePlots
using Random

n = 10
d = 10
κ = 8.0

A = RLinearAlgebra.generate_matrix(n, d, d, κ)
b = randn(n)
iter = 1000
# Row random cyclic projection with block size 4
sol = RLSSolver(
    LinSysVecRowBlockRandCyclic(4),     # Random Cyclic Sampling
    LinSysVecRowBlockProj(),            # Block row projection 
    LSLogFullMA(),                      # Full Moving Average Logger: maintains moving average of residual history
    LSStopMaxIterations(iter),          # Maximum iterations stopping criterion
    nothing                             # System solution
)
x = rsolve(sol, A, b)
plt = lineplot(sol.log.resid_hist, title = "Block Row Random Cyclic", xlabel = "iteration", ylabel = "MA residual")
println(plt)

# Row random sampling with replacement projection with block size 4
sol = RLSSolver(
    LinSysVecRowBlockReplace(4),     # Random Cyclic Sampling
    LinSysVecRowBlockProj(),            # Block row projection 
    LSLogFullMA(),                      # Full Moving Average Logger: maintains moving average of residual history
    LSStopMaxIterations(iter),          # Maximum iterations stopping criterion
    nothing                             # System solution
)

x = rsolve(sol, A, b)
plt = lineplot(sol.log.resid_hist, title = "Block Row Random Replacement", xlabel = "iteration", ylabel = "MA residual")
println(plt)

# Row Gaussian sampling with block size 4
sol = RLSSolver(
    LinSysVecRowBlockGaussian(4),       # Block Gaussian Sampling
    LinSysVecRowBlockProj(),            # Block row projection 
    LSLogFullMA(),                      # Full Moving Average Logger: maintains moving average of residual history
    LSStopMaxIterations(iter),          # Maximum iterations stopping criterion
    nothing                             # System solution
)
x = rsolve(sol, A, b)
plt = lineplot(sol.log.resid_hist, title = "Block Row Gaussian", xlabel = "iteration", ylabel = "MA residual")
println(plt)
