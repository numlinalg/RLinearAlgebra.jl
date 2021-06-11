# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVCDetermCycle

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LSVC Deterministic Cyclic -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysVecColDetermCyclic) == LinSysVecColSampler

    # Test for appropriate cycling of columns
    @test let
        A = rand(3, 10)
        b = rand(3)
        x = rand(10)

        cyc = LinSysVecColDetermCyclic()

        flag = true

        for i = 11:20
            v, nrml, res = RLinearAlgebra.sample(cyc, A, b, x, i)
            flag = flag & (v[i-10] == 1.0)
            flag = flag & (nrml == sum(A[:, i-10].^2))
            flag = flag & (res ≈ dot(A[:,i-10], A * x - b))
        end

        flag
    end
end

end # End module
