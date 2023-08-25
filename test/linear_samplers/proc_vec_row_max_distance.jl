# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRMaxDistance

using Test, RLinearAlgebra

@testset "LSVR Max Distance Selection -- Procedural" begin 
    
    # Verify appropriate super type 
    @test supertype(LinSysVecRowMaxResidual) == LinSysVecRowSampler

    # Test equation selection 
    @test let
        A = vcat(0.5*ones(1,5), ones(9,5))
        x = ones(5)
        b = zeros(10)

        samp = LinSysVecRowMaxDistance()

        α, β = RLinearAlgebra.sample(samp, A, b, x, 1)

        # Given that x = 0, the largest abs residual corresponds to 
        # the last entry of b. So the last row of A should be selected.
        α == A[1,:] && β == b[1]
    end

end

end
