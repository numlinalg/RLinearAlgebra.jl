using RLinearAlgebra
using LinearAlgebra
using UnicodePlots

n = 10
d = 10
κ = 8.0

A = RLinearAlgebra.generate_matrix(n, d, d, κ)
b = randn(n)


sol = LinearSolver(TypeRPM())
x = rsolve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "Default RPM", xlabel = "iteration", ylabel = "residual")
println(plt)

sol = LinearSolver(TypeRPM(SamplerKaczmarzCYC()))
x = rsolve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "Cyclic Kaczmarz", xlabel = "iteration", ylabel = "residual")
println(plt)

sol = LinearSolver(TypeRPM(SamplerKaczmarzCYC(), ProjectionLowCore()))
x = rsolve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "Cyclic with Low Core", xlabel = "iteration", ylabel = "residual")
println(plt)

sol = LinearSolver(TypeRPM(SamplerKaczmarzCYC(), ProjectionFullCore()))
x = rsolve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "Cyclic with Full Core", xlabel = "iteration", ylabel = "residual")
println(plt)


sol = LinearSolver(TypeRPM(SamplerMotzkin()))
x = rsolve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "Motzkin", xlabel = "iteration", ylabel = "residual")
println(plt)

sol = LinearSolver(TypeRPM(SamplerMotzkin(), ProjectionLowCore()))
x = rsolve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "Motzkin low core", xlabel = "iteration", ylabel = "residual")
println(plt)

sol = LinearSolver(TypeRPM(SamplerMotzkin(), ProjectionFullCore()))
x = rsolve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "Motzkin full core", xlabel = "iteration", ylabel = "residual")
println(plt)

sol = LinearSolver(TypeRPM(SamplerGaussSketch()))
x = rsolve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "Gauss", xlabel = "iteration", ylabel = "residual")
println(plt)

sol = LinearSolver(TypeRPM(SamplerGaussSketch(), ProjectionLowCore()))
x = rsolve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "Gauss Low Core", xlabel = "iteration", ylabel = "residual")
println(plt)

sol = LinearSolver(TypeRPM(SamplerGaussSketch(), ProjectionFullCore()))
x = rsolve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "Gauss Full Core", xlabel = "iteration", ylabel = "residual")
println(plt)


samp = SamplerMotzkin()
samp.sampled = true
sol = LinearSolver(TypeRPM(samp))
x = rsolve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "RPM Convergence", xlabel = "iteration", ylabel = "residual")
println(plt)



println("Plot collection of samplers")
plt = lineplot(0:500, zeros(501), width = 100, height = 20, xlabel = "iteration", ylabel = "residual",
               ylim = [0, norm(b)])
samplers = RPMSamplers()
for s in samplers
    solt = LinearSolver(TypeRPM(s))
    xx = rsolve(solt, A, b)
    lineplot!(plt, solt.log.residual_hist)
end
println(plt)

println("Randomized Gauss seidel")
sol = LinearSolver(TypeRGS())
x = rsolve(sol, A, b)
plt = lineplot(sol.log.residual_hist, title = "Randomized Gauss Seidel", xlabel = "iteration", ylabel = "residual")
println(plt)
