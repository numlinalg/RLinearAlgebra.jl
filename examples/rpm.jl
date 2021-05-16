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

sol = LinearSolver(TypeRPM(SamplerMotzkin()))
x = solve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "RPM Convergence", xlabel = "iteration", ylabel = "residual")
println(plt)

sol = LinearSolver(TypeRPM(SamplerGaussSketch()))
x = solve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "RPM Convergence", xlabel = "iteration", ylabel = "residual")
println(plt)

samp = SamplerMotzkin()
samp.sampled = true
sol = LinearSolver(TypeRPM(samp))
x = solve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "RPM Convergence", xlabel = "iteration", ylabel = "residual")
println(plt)



println("Plot collection of samplers")
plt = lineplot(0:500, zeros(501), width = 100, height = 20, xlabel = "iteration", ylabel = "residual",
               ylim = [0, norm(b)])
samplers = RPMSamplers()
for s in samplers
    solt = LinearSolver(TypeRPM(s))
    xx = solve(solt, A, b)
    lineplot!(plt, solt.log.residual_hist)
end
println(plt)
