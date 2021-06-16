# This file is part of RLinearAlgebra.jl

using Test

@testset verbose=true "RLinearAlbera.jl" begin
    for file in readlines(joinpath(@__DIR__, "testgroups.txt"))
        include(file * ".jl")
    end
end
