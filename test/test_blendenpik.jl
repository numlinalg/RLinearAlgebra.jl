n = 1000
d = 20
κ = 1e29

A = RLinearAlgebra.generate_matrix(n, d, d, κ)
b = randn(n)

@testset "Blendenpik vs LU" begin
    x_lu = A \ b
    r_lu = A*x_lu - b

    x_bg = copy(x_lu)
    x_bg .= 0.0
    RLinearAlgebra.blendenpick_gauss!(x_bg, A, b);
    r_bg = A*x_bg - b
end

@testset "Call Bledenpik via API" begin
    sol = RLinearAlgebra.LinearSolver(RLinearAlgebra.TypeBlendenpik())
    x = RLinearAlgebra.rsolve(sol, A, b)
    r = A*x - b
end
