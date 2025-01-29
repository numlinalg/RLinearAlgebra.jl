# This file is part of RLinearAlgebra.jl
using Test

# Modify ≈ to have a global tolerance
ATOL = 1e-10
import Base.≈
≈(a::Float64, b::Float64) = isapprox(a, b, atol = ATOL)
# Adjust for tor the sum of multiple elements
≈(a::AbstractArray, b::AbstractArray) = isapprox(a, b, atol = .5 * ATOL * prod(size(a)))

# Include the test for recipes
include("./test_helpers/field_test_macros.jl")

# List all directories that have files to be tested 
directs = joinpath.(@__DIR__,["./", "Approximators/", "Solvers/", "Solvers/Loggers/", 
                              "Solvers/SubSolvers/", "Compressors/"])

@testset verbose=true "RLinearAlbera.jl" begin
    for direct in directs 
        # Obtain all files in the directory
        files_in_direct = readdir(direct)
        # Only test files that end in .jl
        files_to_test = files_in_direct[occursin.(r".jl$", files_in_direct)]
        for file in files_to_test  
            # Make sure you do not call the runtest file otherwise you have infinite recursion
            if file != "runtest.jl"
                include(direct * file)
            end

        end

    end

end
