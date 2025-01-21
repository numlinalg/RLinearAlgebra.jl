# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRPropToNormSampler

using Test, RLinearAlgebra, Random, StatsBase

Random.seed!(1010)

@testset "LSVR Strohmer-Vershynin Sampling -- Procedural" begin

    # Verify appropriate super type
    @test supertype(LinSysVecRowPropToNormSampler) == LinSysVecRowSampler

    # Verify alias
    @test LinSysVecRowPropToNormSampler == LinSysVecRowSVSampler

    # Verify field values
    field_names = [:dist]
    let field_names = field_names
        @test length(fieldnames(LinSysVecRowPropToNormSampler)) == 1
        @test field_names[1] in fieldnames(LinSysVecRowPropToNormSampler)
    end

    # Verify default constructor
    sampler = LinSysVecRowPropToNormSampler()
    @test sampler.dist == [1.0]

    # test distribution is set correctly

    ## set seed for reproducibility
    Random.seed!(1010)

    ## Generate system
    A = randn(10, 5)
    b = randn(10)
    x = randn(5)
    iter = 1

    let A = A, b = b, x = x, iter = iter
        sampler = LinSysVecRowPropToNormSampler()
        output = RLinearAlgebra.sample(sampler, A, b, x, iter)
        @test length(output) == 2
        @test sampler.dist == RLinearAlgebra.frobenius_norm_distribution(A, true)
    end

    # test sampling is done correctly
    let A = A, b = b, x = x, iter = iter
        Random.seed!(1010)
        sampler = LinSysVecRowPropToNormSampler() 
        output = RLinearAlgebra.sample(sampler, A, b, x, iter)
        
        Random.seed!(1010)
        eqn_ind = StatsBase.sample(1:length(sampler.dist), Weights(sampler.dist))
        
        @test A[eqn_ind, :] == output[1]
        @test b[eqn_ind] == output[2]
    end

    # test sampling is done correctly for an arbitrary iteration
    let A = A, b = b, x = x, iter = iter

        ## initialize
        Random.seed!(1010)
        sampler = LinSysVecRowPropToNormSampler() 
        output = RLinearAlgebra.sample(sampler, A, b, x, iter)
        
        ## use sampling function
        Random.seed!(1010)
        output = RLinearAlgebra.sample(sampler, A, b, x, iter + rand(1:100))
        
        ## reset seed and RLinearAlgebra.sample from distribution
        Random.seed!(1010)
        eqn_ind = StatsBase.sample(1:length(sampler.dist), Weights(sampler.dist))

        ## check correctness
        @test sampler.dist == RLinearAlgebra.frobenius_norm_distribution(A, true)
        @test A[eqn_ind, :] == output[1]
        @test b[eqn_ind] == output[2] 
    end
end

end
