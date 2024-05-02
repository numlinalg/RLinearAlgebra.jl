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
    LinSysVecColBlockRandCyclic(4),     # Random Cyclic Sampling
    LinSysVecColBlockProj(),            # Block column projection 
    LSLogFullMA(),                      # Full Moving Average Logger: maintains moving average of residual history
    LSStopMaxIterations(iter),          # Maximum iterations stopping criterion
    nothing                             # System solution
)
x = rsolve(sol, A, b)
plt = lineplot(sol.log.resid_hist, title = "Block Col Random Cyclic", xlabel = "iteration", ylabel = "MA gradient")
println(plt)

# Col Gaussian sampling with block size 4
sol = RLSSolver(
    LinSysVecColBlockGaussian(4),       # Block Gaussian Sampling
    LinSysVecColBlockProj(),            # Block column projection 
    LSLogFullMA(),                      # Full Moving Average Logger: maintains moving average of residual history
    LSStopMaxIterations(iter),          # Maximum iterations stopping criterion
    nothing                             # System solution
)
x = rsolve(sol, A, b)
plt = lineplot(sol.log.resid_hist, title = "Block Col Gaussian", xlabel = "iteration", ylabel = "MA gradient")
println(plt)
