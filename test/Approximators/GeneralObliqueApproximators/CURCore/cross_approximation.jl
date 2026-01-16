module CrossApproximation 
using Test, RLinearAlgebra, LinearAlgebra, SparseArrays

mutable struct TestCUR <: Approximator
    rank::Int64
    oversample::Int64
end

TestCUR() = TestCUR(2,1)

mutable struct TestCURRecipe <: ApproximatorRecipe
    n_row_vecs::Int64
    n_col_vecs::Int64
    col_idx::AbstractVector
    row_idx::AbstractVector
end

function complete_approximator(approx::TestCUR, A::AbstractMatrix)
    n_col_vecs = approx.rank
    n_row_vecs = approx.rank + approx.oversample 
    col_idx = 1:n_col_vecs
    row_idx = 1:n_row_vecs
end

@testset "CrossApproximation" begin
    @testset "CrossApproximation" begin
        @test supertype(CrossApproximation) == Core

        # test the fieldnames and types
        @test fieldnames(CrossApproximation) == ()
        @test fieldtypes(CrossApproximation) == ()

        # test constructor
        core = CrossApproximation()

        @test typeof(core) == CrossApproximation

    end

    @testset "CrossApproximation Recipe" begin
        supertype(CrossApproximationRecipe) == CoreRecipe

        # test the fieldnames and types
        @test fieldnames(CrossApproximationRecipe) == (
            :n_row_vecs, :n_col_vecs, :core, :core_view, :qr_decomp
        )
        @test fieldtypes(CrossApproximationRecipe) == (
            Int64, Int64, core, core_view, qr_decomp
        )
    end

    @testset "CrossApproximation: Complete Core" begin
        # test with matrix
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = rand(n_rows, rank) * rand(rank, n_cols)
            approx = TestCUR(rank, oversample) 
            core = CrossApproximation()

            # test the complete core function
            recipe = complete_core(approx, core, A)
            @test size(recipe) == (rank, rank + oversample)
            # check that the core is identity
            @test sum(diag(recipe.core) .== 1) == rank
            @test size(approx.core_view) == (2, 2)
            @test typeof(approx.qr_decomp) <: QR
        end

        # test with sparse matrix
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = sprand(n_rows, rank, .9) * sprand(rank, n_cols, .9)
            approx = TestCUR(rank, oversample) 
            core = CrossApproximation()

            # test the complete core function
            recipe = complete_core(approx, core, A)
            @test size(recipe) == (rank, rank + oversample)
            # check that the core is identity
            @test sum(diag(recipe.core) .== 1) == rank
            @test size(approx.core_view) == (2, 2)
            @test typeof(approx.qr_decomp) <: QR
        end

    end
    #################################################################
    # It remains to test the update core and multiplication functions,
    # in addition for the multiplication you must also define an adjoint 
    # for the core, the size function for the core, and the multiplication 
    # function for the adjoint of the core. After completing this you should 
    # do the same for the optimal core then for the CUR
    ######################################################################
end

end