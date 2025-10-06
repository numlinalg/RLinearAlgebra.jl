module LUPP_tests
using Test
using ..FieldTest
using ..ApproxTol
using RLinearAlgebra
import LinearAlgebra: mul!

@testset "LUPP Tests" begin
    @testset "LUPP" begin
        supertype(LUPP) == Selector
        
        # test fieldnames and types
        fieldnames(LUPP) == ()
        fieldtypes(LUPP) == ()
        
        # Test Constructor
        let 
            sel = LUPP()
            @test typeof(sel) == LUPP 
        end

    end

    @testset "LUPPRecipe" begin
        supertype(LUPPRecipe) == SelectorRecipe
        # test fieldnames and types
        fieldnames(LUPPRecipe) == ()
        fieldtypes(LUPPRecipe) == ()


        # Test Constructor
        let 
            sel = LUPPRecipe()
            @test typeof(sel) == LUPPRecipe 
        end

    end

    @testset "LUPP: Complete Selector" begin
        let sel = LUPP()
            sel_rec = complete_selector(sel)
            @test typeof(sel_rec) == LUPPRecipe
        end

    end

    @testset "LUPP: Update Selector" begin
        let sel_rec = complete_selector(LUPP())
            update_selector!(sel_rec)
            @test typeof(sel_rec) == LUPPRecipe 
        end

    end

    @testset "LUPP: Select Indices" begin
        A = [0 1 0;
             0 0 2;
             3 0 0]
        # test selecting only one index
        let A = deepcopy(A),
            idx = zeros(Int64, 3),
            start_idx = 2,
            n_idx = 1
            
            select_indices!(idx, LUPPRecipe(), A, n_idx, start_idx)
            @test idx[2] == 2
        end

        # test selecting three indices
        let A = deepcopy(A),
            idx = zeros(Int64, 3),
            start_idx = 1,
            n_idx = 3

            select_indices!(idx, LUPPRecipe(), A, n_idx, start_idx)
            @test idx == [2; 3; 1]
        end

        # test the error checking
        let A = A,
            idx = zeros(Int64, 3),
            start_idx = 1,
            n_idx = 4

            @test_throws DimensionMismatch select_indices!(
                            idx, 
                            LUPPRecipe(), 
                            A, 
                            n_idx, 
                            start_idx
                ) 
        end

        let A = A,
            idx = zeros(Int64, 3),
            start_idx = 3,
            n_idx = 2

            @test_throws DimensionMismatch select_indices!(
                            idx, 
                            LUPPRecipe(), 
                            A, 
                            n_idx, 
                            start_idx
                ) 
        end

    end

end

end
