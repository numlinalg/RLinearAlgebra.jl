n = 1000
d = 20
κ = 1e29

A = RLinearAlgebra.generate_matrix(n, d, d, κ)
b = randn(n)

@testset "Blendenpik vs LU" begin

    x_lu = A \ b
    r_lu = A*x_lu - b

    x_bg = RLinearAlgebra.Solvers.blendenpick_gauss(A, b);
    r_bg = A*x_bg - b

    printstyled("""
    SOLVING: $n × $d full column rank least squares problem.

    LU SOLVER:
    \tResidual Norm: $(norm(r_lu))
    \tNormal System Residual Norm: $(norm(A'*r_lu))

    BLENDENPIK_GAUSS:
    \tResidual Norm: $(norm(r_bg))
    \tNormal System Residual Norm: $(norm(A'*r_bg))
    """
    )
end

@testset "Call Bledenpik via API" begin
    sol = RLinearAlgebra.Solvers.LinearSolver(RLinearAlgebra.Solvers.TypeBlendenpik())
    x = RLinearAlgebra.Solvers.solve(sol, A, b)
    r = A*x - b
    println("Residual norm: ", norm(A'*r))
end
