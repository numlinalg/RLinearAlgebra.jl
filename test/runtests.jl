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
            if file != "runtests.jl"
                include(direct * file)
            end

        end

    end

end
