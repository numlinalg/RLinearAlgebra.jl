module CrossApproximation 
using Test, RLinearAlgebra, LinearAlgebra, SparseArrays

# Write test selector recipe 
mutable struct TestSelectorRecipe <: SelectorRecipe end

function select_indices!(
    idx::Vector{Int64}, 
    recipe::TestSelectorRecipe,
    A::AbstractMatrix,
    n::Int64,
    offset::Int64
)
    idx[1:n] = 1:n
end

# write test CUR structure
mutable struct TestCUR <: Approximator
    rank::Int64
    oversample::Int64
end

TestCUR() = TestCUR(2,1)

# Create test CUR approximator
mutable struct TestCURRecipe{CR <: CrossApproximationRecipe} <: ApproximatorRecipe
    n_row_vecs::Int64
    n_col_vecs::Int64
    col_selector::SelectorRecipe
    row_selector::SelectorRecipe
    col_idx::AbstractVector
    row_idx::AbstractVector
    C::AbstractMatrix
    R::AbstractMatrix
    U::CR
end

function complete_approximator(approx::TestCUR, A::AbstractMatrix)
    n_col_vecs = approx.rank
    n_row_vecs = approx.rank + approx.oversample 
    col_idx = ones(n_col_vecs)
    row_idx = ones(n_row_vecs)
    C = zeros(size(A, 1), n_col_vecs)
    R = zeros(n_row_vecs, size(A, 2))
    U = complete_core(TestCUR, CrossApproximation(), A)
    return TestCURRecipe{CrossApproximationRecipe}(
        n_row_vecs,
        n_col_vecs,
        TestSelectorRecipe(),
        TestSelectorRecipe(),
        col_idx,
        row_idx,
        C,
        R,
        U
    )
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
            @test typeof(approx.qr_decomp) <: LinearAlgebra.QRCompactWY
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
            @test typeof(approx.qr_decomp) <: SparseArrays.SPQR.QRSparse
        end

    end
    
    @testset "CrossApproximation: Update Core" begin
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
            approx_recipe = complete_approximator(approx, A)
            # change indices selected to be first n
            approx_recipe.col_idx = 1:approx_recipe.n_col_vecs
            approx_recipe.row_idx = 1:approx_recipe.n_row_vecs
            # update the core recipe
            update_recipe!(recipe, approx_recipe, A)
            # check that a QR decomposition is stored of with correct sizes
            @test size(recipe) == (rank, rank + oversample)
            # check that the core is the correct QR decomposition
            @test typeof(approx.qr_decomp) <: LinearAlgebra.QRCompactWY
            Q, R = qr(A[approx_recipe.col_idx, approx_recipe.row_idx])
            @test approx.qr_decomp.Q == Q
            @test approx.qr_decomp.R == R
        end

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
            approx_recipe = complete_approximator(approx, A)
            # change indices selected to be first n
            approx_recipe.col_idx = 1:approx_recipe.n_col_vecs
            approx_recipe.row_idx = 1:approx_recipe.n_row_vecs
            # update the core recipe
            update_recipe!(recipe, approx_recipe, A)
            # check that a QR decomposition is stored of with correct sizes
            @test size(recipe) == (rank, rank + oversample)
            # check that the core is the correct QR decomposition
            @test typeof(approx.qr_decomp) <: SparseArrays.SPQR.QRSparse
            Q, R = qr(A[approx_recipe.col_idx, approx_recipe.row_idx])
            @test approx.qr_decomp.Q == Q
            @test approx.qr_decomp.R == R
        end

    end

    @testset "CrossApproximation: rapproximate" begin
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = rand(n_rows, rank) * rand(rank, n_cols)
            
            approx = TestCUR(rank, oversample) 
            approx_recipe = complete_approximator(approx, A)
            rapproximate!(approx_recipe, A)
            col_idx = approx_recipe.n_col_vecs
            row_idx = approx_recipe.n_row_vecs
            # test that the cur approximation corresponds to the first 2 columns and 
            # first 3 rows
            @test approx_recipe.col_idx == 1:col_idx
            @test approx_recipe.row_idx == 1:row_idx
            @test approx_recipe.C == A[:, 1:col_idx]
            @test approx_recipe.R == A[1:row_idx, :]
            @test typeof(approx_recipe.U) == CrossApproximationRecipe
            @test approx_recipe.U.qr_decomp == qr!(A[1:row_idx, 1:col_idx])
            @test size(approx_recipe.U) == (row_idx, col_idx)
        end

        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = sprand(n_rows, rank, .9) * sprand(rank, n_cols, .9)

            approx = TestCUR(rank, oversample) 
            approx_recipe = complete_approximator(approx, A)
            rapproximate!(approx_recipe, A)
            col_idx = approx_recipe.n_col_vecs
            row_idx = approx_recipe.n_row_vecs
            # test that the cur approximation corresponds to the first 2 columns and 
            # first 3 rows
            @test approx_recipe.col_idx == 1:col_idx
            @test approx_recipe.row_idx == 1:row_idx
            @test approx_recipe.C == A[:, 1:col_idx]
            @test approx_recipe.R == A[1:row_idx, :]
            @test typeof(approx_recipe.U) == CrossApproximationRecipe
            @test approx_recipe.U.qr_decomp == qr!(A[1:row_idx, 1:col_idx])
            @test size(approx_recipe.U) == (row_idx, col_idx)
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