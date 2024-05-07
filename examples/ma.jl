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

sol_approximate = RLSSolver(
    LinSysVecRowRandCyclic(),   # Random Cyclic Sampling
    LinSysVecRowProjPO(),       # Partially Orthogonalized Row Projection, 5 vector memory
    LSLogMA(),                  # Moving Average Logger: maintains moving average of residual history
    LSStopMaxIterations(iter),  # Maximum iterations stopping criterion
    nothing                     # System solution
)
sol_exact = RLSSolver(
    LinSysVecRowRandCyclic(),   # Random Cyclic Sampling
    LinSysVecRowProjPO(),       # Partially Orthogonalized Row Projection, 5 vector memory
    LSLogMA(true_res = true),   # Moving Average Logger: maintains moving average of exact residual history
    LSStopMaxIterations(iter),  # Maximum iterations stopping criterion
    nothing                     # System solution
)
#Get results from exact moving average
Random.seed!(12321)
x1 = rsolve(sol_exact, A, b)
#Get results from approximate moving average
Random.seed!(12321)
x2 =  rsolve(sol_approximate, A, b)
#plot the results
bounds = get_uncertainty(sol_approximate.log)
plt = lineplot(1:length(bounds[2]), 
               [bounds[2] bounds[1] bounds[3]], 
               xlabel = "Iteration", 
               ylabel = "Estimate", 
               title = "Evolution of MA over RandCyclic", 
               name = ["Upper", "Estimate", "Lower"])
println(plt)


plt = lineplot(2:length(bounds[2]), 
               abs.(bounds[1][2:end] - sol_exact.log.resid_hist[2:end]),
               yscale = :log10,
               xlabel = "Iteration", 
               ylabel = "Estimate", 
               title = "Difference between approximate and exact MA", 
               name = "Difference")
println(plt)
