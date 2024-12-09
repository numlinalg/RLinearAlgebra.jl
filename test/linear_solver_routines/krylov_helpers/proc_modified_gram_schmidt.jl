# This file is part of RLinearAlgebra.jl
# Date: 12/3/2024
# Author: Christian Varner
# Purpose: Test cases for modified gram schmidt in krylov helpers

module ProceduralMGS

using Test, RLinearAlgebra, LinearAlgebra, Random

@testset "MGS -- Procedural" begin

    # set testing seed
    Random.seed!(1010)

    # test that functions are defined
    @test isdefined(RLinearAlgebra, :mgs!)

    ##################################
    # test first method
    ##################################
    dim = 100
    nbasis = 1
    q_original = randn(dim)
    q_copy = copy(q_original)
    basis = randn(dim, nbasis)
    basis ./= norm(basis)

    h = RLinearAlgebra.mgs!(q_copy, basis)

    for i in 1:nbasis
        @test dot(q_copy, basis[:, i]) <= eps() * dim
        q_original .-= h[i] .* basis[:, i]
    end
    @test norm(q_original - q_copy) <= eps() * dim 

    # continuously add to previous basis
    k = rand(collect(1:10))
    for i in 1:k
        q_copy ./= norm(q_copy)
        basis = hcat(basis, q_copy)
        nbasis += 1

        q_original = randn(dim)
        q_copy = copy(q_original)

        h = RLinearAlgebra.mgs!(q_copy, basis)

        for i in 1:nbasis
            @test dot(q_copy, basis[:, i]) <= eps() * dim
            q_original .-= h[i] .* basis[:, i]
        end
        
        @test norm(q_original - q_copy) <= eps() * dim
    end

    ##################################
    # test second method
    ##################################
    dim = 100
    nbasis = 1
    q_original = randn(dim)
    q_copy = copy(q_original)
    basis = randn(dim, nbasis)
    basis ./= norm(basis)
    h = zeros(1, 1)


    RLinearAlgebra.mgs!(q_copy, view(h, :, 1), basis)

    for i in 1:nbasis
        @test dot(q_copy, basis[:, i]) <= eps() * dim
        q_original .-= h[i, i] .* basis[:, i]
    end
    @test norm(q_original - q_copy) <= eps() * dim

    # continuously add to previous basis
    k = rand(collect(1:10))
    for i in 1:k
        q_copy ./= norm(q_copy)
        basis = hcat(basis, q_copy)
        nbasis += 1

        q_original = randn(dim)
        q_copy = copy(q_original)
        h = zeros(nbasis, 1)

        RLinearAlgebra.mgs!(q_copy, view(h, :, 1), basis)

        for i in 1:nbasis
            @test dot(q_copy, basis[:, i]) <= eps() * dim
            q_original .-= h[i, 1] .* basis[:, i]
        end
        
        @test norm(q_original - q_copy) <= eps() * dim
    end
end


end
