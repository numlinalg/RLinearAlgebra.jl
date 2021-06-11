# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRProjPO

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "LSVR Projection Partial Orthogonal -- Procedural" begin

    # Supertype
    @test supertype(LinSysVecRowProjPO) == LinSysVecRowProjection

    
end

end # End Module
