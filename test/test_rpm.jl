n = 10
d = 10
κ = 8.0

A = RLinearAlgebra.generate_matrix(n, d, d, κ)
b = randn(n)
@testset "Call basic RPM via API" begin
    sol = RLinearAlgebra.Solvers.LinearSolver(RLinearAlgebra.Solvers.TypeRPM())
    x = RLinearAlgebra.Solvers.solve(sol, A, b)
    x_lu = A\b
    println("Norm difference: ", norm(x - x_lu))
end
