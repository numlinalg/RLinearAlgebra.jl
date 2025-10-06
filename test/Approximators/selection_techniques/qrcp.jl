module QRCP_tests
using Test
using ..FieldTest
using ..ApproxTol
using RLinearAlgebra
import LinearAlgebra: mul!

@testset "QRCP Tests" begin
    @testset "QRCP" begin
        supertype(QRCP) == Selector
        
        # test fieldnames and types
        fieldnames(QRCP) == ()
        fieldtypes(QRCP) == ()
        
        # Test Constructor
        let 
            sel = QRCP()
            @test typeof(sel) == QRCP 
        end

    end

    @testset "QRCPRecipe" begin
        supertype(QRCPRecipe) == SelectorRecipe
        # test fieldnames and types
        fieldnames(QRCPRecipe) == ()
        fieldtypes(QRCPRecipe) == ()


        # Test Constructor
        let 
            sel = QRCPRecipe()
            @test typeof(sel) == QRCPRecipe 
        end

    end

    @testset "QRCP: Complete Selector" begin
        let sel = QRCP()
            sel_rec = complete_selector(sel)
            @test typeof(sel_rec) == QRCPRecipe
        end

    end

    @testset "QRCP: Update Selector" begin
        let sel_rec = complete_selector(QRCP())
            update_selector!(sel_rec)
            @test typeof(sel_rec) == QRCPRecipe 
        end

    end

    @testset "QRCP: Select Indices" begin
        A = [0 1 0;
             0 0 2;
             3 0 0]
        # test selecting only one index
        let A = deepcopy(A),
            idx = zeros(Int64, 3),
            start_idx = 2,
            n_idx = 1
            
            select_indices!(idx, QRCPRecipe(), A, n_idx, start_idx)
            @test idx[2] == 1
        end

        # test selecting three indices
        let A = deepcopy(A),
            idx = zeros(Int64, 3),
            start_idx = 1,
            n_idx = 3

            select_indices!(idx, QRCPRecipe(), A, n_idx, start_idx)
            @test idx == [1; 3; 2]
        end

        # test the error checking
        let A = A,
            idx = zeros(Int64, 3),
            start_idx = 1,
            n_idx = 4

            @test_throws DimensionMismatch select_indices!(
                            idx, 
                            QRCPRecipe(), 
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
                            QRCPRecipe(), 
                            A, 
                            n_idx, 
                            start_idx
                ) 
        end

    end

end

end
