# This file is part of RLinearAlgebra.jl

module ProceduralTestLinSysSampler

using Test, RLinearAlgebra, Random

@testset "Linear System Sampler Abstractions" begin

    @test LinSysSketch == LinSysSampler #Check alias
    @test LinSysSelect == LinSysSampler #Check alias

    @test supertype(LinSysVecRowSampler) == LinSysSampler #Check parent type
    @test LinSysVecRowSketch == LinSysVecRowSampler #Check alias
    @test LinSysVecRowSelect == LinSysVecRowSampler #Check alias

    @test supertype(LinSysVecColSampler) == LinSysSampler #Check parent type
    @test LinSysVecColSketch == LinSysVecColSampler #Check alias
    @test LinSysVecColSelect == LinSysVecColSampler #Check alias

    @test supertype(LinSysBlkRowSampler) == LinSysSampler #Check parent type
    @test LinSysBlkRowSketch == LinSysBlkRowSampler #Check alias
    @test LinSysBlkRowSelect == LinSysBlkRowSampler #Check alias

    @test supertype(LinSysBlkColSampler) == LinSysSampler #Check parent type
    @test LinSysBlkColSketch == LinSysBlkColSampler #Check alias
    @test LinSysBlkColSelect == LinSysBlkColSampler #Check alias
end


end
