# This file is part of RLinearAlgebra.jl

module ProceduralTestRLSSolver

using Test, RLinearAlgebra, Random

@testset "Randomized Linear Solver -- Procedural" begin

    # Check if solver encapsulation is defined
    @test @isdefined RLSSolver

    # Verify required fields
    @test :sampler in fieldnames(RLSSolver)
    @test :routine in fieldnames(RLSSolver)
    @test :log in fieldnames(RLSSolver)
    @test :stop in fieldnames(RLSSolver)
    @test :x in fieldnames(RLSSolver)

end

end # End module
