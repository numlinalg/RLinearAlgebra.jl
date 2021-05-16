using RLinearAlgebra
using LinearAlgebra
using UnicodePlots

n = 10
d = 10
κ = 8.0

A = RLinearAlgebra.generate_matrix(n, d, d, κ)
b = randn(n)


sol = LinearSolver(TypeRPM())
x = solve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "RPM Convergence", xlabel = "iteration", ylabel = "residual")
println(plt)

sol = LinearSolver(TypeRPM(SamplerKaczmarzCYC()))
x = solve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "RPM Convergence", xlabel = "iteration", ylabel = "residual")
println(plt)
