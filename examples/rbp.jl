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
    LinSysBlkRowRandCyclic(4),          # Random Cyclic Sampling
    LinSysBlkRowProj(),                 # Block row projection 
    LSLogMA(),                          # Full Moving Average Logger: maintains moving average of residual history
    LSStopMaxIterations(iter),          # Maximum iterations stopping criterion
    nothing                             # System solution
)
x = rsolve(sol, A, b)
plt = lineplot(sol.log.resid_hist, title = "Block Row Random Cyclic", xlabel = "iteration", ylabel = "MA residual")
println(plt)

# Row random sampling with replacement projection with block size 4
sol = RLSSolver(
    LinSysBlkRowReplace(4),             # Random Cyclic Sampling
    LinSysBlkRowProj(),                 # Block row projection 
    LSLogMA(),                          # Full Moving Average Logger: maintains moving average of residual history
    LSStopMaxIterations(iter),          # Maximum iterations stopping criterion
    nothing                             # System solution
)

x = rsolve(sol, A, b)
plt = lineplot(sol.log.resid_hist, title = "Block Row Random Replacement", xlabel = "iteration", ylabel = "MA residual")
println(plt)

# Row Gaussian sampling with block size 4
sol = RLSSolver(
    LinSysBlkRowGaussSampler(4),        # Block Gaussian Sampling
    LinSysBlkRowProj(),                 # Block row projection 
    LSLogMA(),                          # Full Moving Average Logger: maintains moving average of residual history
    LSStopMaxIterations(iter),          # Maximum iterations stopping criterion
    nothing                             # System solution
)
x = rsolve(sol, A, b)
plt = lineplot(sol.log.resid_hist, title = "Block Row Gaussian", xlabel = "iteration", ylabel = "MA residual")
println(plt)
