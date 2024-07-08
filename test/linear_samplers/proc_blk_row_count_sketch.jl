# This file is part of RLinearAlgebra.jl
# Date: 07/08/2024
# Author: Christian Varner
# Purpose: Test the sampling procedure CountSketch from
# src/linear_samplers/block_row_count_sketch.jl

module ProceduralTestLSBRCountSketch

using Test, RLinearAlgebra, Random, LinearAlgebra

Random.seed!(1010)

@testset "LSBR Count Sketch -- Procedural" begin

    ## Verify appropriate super type
    @test supertype(LinSysBlkRowCountSketch) <: LinSysVecRowSelect
    
    # test first constructor
    sampler = LinSysBlkRowCountSketch(10,nothing,nothing,nothing)
    @test sampler.size == 10
    @test isnothing(sampler.labels)
    @test isnothing(sampler.signs)
    @test isnothing(sampler.S)

    sampler = LinSysBlkRowCountSketch(10,zeros(5), zeros(6), zeros(10,10))
    @test sampler.size == 10
    @test sampler.labels == zeros(5)
    @test sampler.signs == zeros(6)
    @test sampler.S == zeros(10,10)

    # test second constructor
    sampler = LinSysBlkRowCountSketch(225)
    @test sampler.size == 225
    @test isnothing(sampler.labels)
    @test isnothing(sampler.signs)
    @test isnothing(sampler.S) 

    # test default constructor
    sampler = LinSysBlkRowCountSketch()
    @test sampler.size == 1
    @test isnothing(sampler.labels)
    @test isnothing(sampler.signs)
    @test isnothing(sampler.S) 

    ## test procedure and output of method
    A = randn(10,5)
    b = randn(10)

    sampler = LinSysBlkRowCountSketch(6)

    # initial iteration
    S,SA,res=RLinearAlgebra.sample(sampler,A,b,zeros(5),1)

    # sketch sizes are correct
    @test size(S) == (6,10)
    @test size(SA) == (6,5)

    # reported sketch matrix and sketched linear systems correct
    @test norm(SA-S*A) < eps()*1e2
    @test norm(res - (S*A*zeros(5)-S*b)) < eps()*1e2

    # Only one label for each row (one 1 or -1)
    @test typeof(S) == Matrix{Int64}
    for j in 1:10
        # sum elements in each row (which are integers if the above test passed)
        s = 0
        for i in 1:size(S)[1]
            s += abs(S[i,j])
        end
        @test s == 1 
    end
end

end # End module

