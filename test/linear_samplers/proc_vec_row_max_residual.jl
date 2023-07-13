# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRMaxResidual

using Test, RLinearAlgebra

@testset "LSVR Max Residual Selection -- Procedural" begin 
    
    # Verify appropriate super type 
    @test supertype(LinSysVecRowMaxResidual) == LinSysVecRowSampler

    # Test equation selection 
    @test let
        A = rand(10,5)
        x = zeros(5)
        b = -collect(1:10)

        samp = LinSysVecRowMaxResidual()

        α, β = RLinearAlgebra.sample(samp, A, b, x, 1)

        # Given that x = 0, the largest abs residual corresponds to 
        # the last entry of b. So the last row of A should be selected.
        α == A[end,:] && β == b[end]
    end

end

end
