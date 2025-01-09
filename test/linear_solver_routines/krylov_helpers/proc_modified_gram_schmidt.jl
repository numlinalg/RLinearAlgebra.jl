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
    let dim = 100
        # continuously add to previous basis
        nbasis = 1
        q_original = randn(dim); q_copy = copy(q_original);
        basis = randn(dim, nbasis); basis ./= norm(basis);
        k = rand(2:10)
        for i in 1:k
            # get orthogonalize q_copy against basis
            h = RLinearAlgebra.mgs!(q_copy, basis)

            # test orthogonality and that mgs did the process correctly
            for i in 1:nbasis
                @test dot(q_copy, basis[:, i]) <= eps() * dim
                q_original .-= h[i] .* basis[:, i]
            end
            
            # q_original should be q_copy if h is saved correctly
            @test norm(q_original - q_copy) <= eps() * dim
            
            # expand basis
            q_copy ./= norm(q_copy)
            basis = hcat(basis, q_copy)
            nbasis += 1
            
            # get new vectors
            q_original = randn(dim)
            q_copy = copy(q_original)
        end
    end

    ##################################
    # test second method
    ##################################
    let dim = 100
        
        # storage vectors
        nbasis = 1
        q_original = randn(dim)
        q_copy = copy(q_original)
        basis = randn(dim, nbasis)
        basis ./= norm(basis)
        h = zeros(1, 1)
       
        # continuously add to basis
        k = rand(2:10)
        for i in 1:k
            RLinearAlgebra.mgs!(q_copy, view(h, :, 1), basis)

            for i in 1:nbasis
                @test dot(q_copy, basis[:, i]) <= eps() * dim
                q_original .-= h[i, 1] .* basis[:, i]
            end

            @test norm(q_original - q_copy) <= eps() * dim
            
            q_copy ./= norm(q_copy)
            basis = hcat(basis, q_copy)
            nbasis += 1
            
            q_original = randn(dim)
            q_copy = copy(q_original)
            h = zeros(nbasis, 1)
        end
    end
end


end
