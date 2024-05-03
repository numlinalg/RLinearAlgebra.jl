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
# Col random cyclic projection with block size 4
sol = RLSSolver(
    LinSysBlkColRandCyclic(4),          # Random Cyclic Sampling
    LinSysBlkColProj(),                 # Block column projection 
    LSLogFullMA(),                      # Full Moving Average Logger: maintains moving average of residual history
    LSStopMaxIterations(iter),          # Maximum iterations stopping criterion
    nothing                             # System solution
)
x = rsolve(sol, A, b)
plt = lineplot(sol.log.resid_hist, title = "Block Col Random Cyclic", xlabel = "iteration", ylabel = "MA gradient")
println(plt)

# Col random sampling with replacement projection with block size 4
sol = RLSSolver(
    LinSysBlkColReplace(4),             # Random Cyclic Sampling
    LinSysBlkColProj(),                 # Block column projection 
    LSLogFullMA(),                      # Full Moving Average Logger: maintains moving average of residual history
    LSStopMaxIterations(iter),          # Maximum iterations stopping criterion
    nothing                             # System solution
)
x = rsolve(sol, A, b)
plt = lineplot(sol.log.resid_hist, title = "Block Col Random Replacement", xlabel = "iteration", ylabel = "MA gradient")
println(plt)

# Col Gaussian sampling with block size 4
sol = RLSSolver(
    LinSysBlkColGaussSampler(4),        # Block Gaussian Sampling
    LinSysBlkColProj(),                 # Block column projection 
    LSLogFullMA(),                      # Full Moving Average Logger: maintains moving average of residual history
    LSStopMaxIterations(iter),          # Maximum iterations stopping criterion
    nothing                             # System solution
)
x = rsolve(sol, A, b)
plt = lineplot(sol.log.resid_hist, title = "Block Col Gaussian", xlabel = "iteration", ylabel = "MA gradient")
println(plt)
