# This file is part of RLinearAlgebra.jl

for file in readlines(joinpath(@__DIR__, "testgroups.txt"))
    include(file * ".jl")
end
