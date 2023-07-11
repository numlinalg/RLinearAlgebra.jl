# This file is part of RLinearAlgebra.jl

module ProceduralTestLSVRUnifSparseSampler

using Test, RLinearAlgebra, Random

Random.seed!(1010)

@testset "LSVR Uniform Sparse Sampling -- Procedural" begin

        #Verify appropriate super type
        @test supertype(LinSysVecRowSparseUnifSampler) == LinSysVecRowSampler

        #Test constructor throws error if sparsity greater than one
        @test try 
            samp = LinSysVecRowSparseUnifSampler(1.1)
            false
        catch e
            typeof(e)==DomainError ? true : false
        end

        #Test constructor throws error if sparsity equals one
        @test try 
            samp = LinSysVecRowSparseUnifSampler(1.0)
            false 
        catch e
            typeof(e)==DomainError ? true : false
        end

        #Test constructor throws error if sparsity less than 0
        @test try 
            samp = LinSysVecRowSparseUnifSampler(-0.2)
            false
        catch e
            typeof(e)==DomainError ? true : false
        end
        

        #Test constructor throws error if sparsity is 0
        @test try 
            samp = LinSysVecRowSparseUnifSampler(0.0)
            false
        catch e
            typeof(e)==DomainError ? true : false
        end

        # Test sampling method 
        @test let
            A = ones(10,5)
            b = ones(10)
            x = rand(5)

            samp = LinSysVecRowSparseUnifSampler() #Default 0.2 sparsity
            α, β = RLinearAlgebra.sampl(samp, A, b, x, 1)

            true
        end

end

end 