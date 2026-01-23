module CrossApproximation 
using Test, RLinearAlgebra, LinearAlgebra, SparseArrays

# Write test selector recipe 
mutable struct TestSelector <: Selector end 
mutable struct TestSelectorRecipe <: SelectorRecipe end

function RLinearAlgebra.complete_selector(TestSelector::Selector, A::AbstractArray)
    return TestSelectorRecipe()
end

function RLinearAlgebra.select_indices!(
    idx::Vector{Int64}, 
    recipe::TestSelectorRecipe,
    A::AbstractMatrix,
    n::Int64,
    offset::Int64
)
    idx[1:n] = 1:n
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
            approx = CUR(
                rank, 
                oversample = oversample, 
                selector_cols = TestSelector()
            ) 
            core = CrossApproximation()

            # test the complete core function
            recipe = complete_core(approx, core, A)
            @test size(recipe) == (rank, rank + oversample)
            # check that the core is identity
            @test sum(diag(recipe.core) .== 1) == rank
            @test size(recipe.core_view) == (2, 2)
            @test typeof(recipe.qr_decomp) <: LinearAlgebra.QRCompactWY
        end

        # test with sparse matrix
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = sprand(n_rows, rank, .9) * sprand(rank, n_cols, .9)
            approx = CUR(
                rank, 
                oversample = oversample, 
                selector_cols = TestSelector()
            ) 
            core = CrossApproximation()

            # test the complete core function
            recipe = complete_core(approx, core, A)
            @test size(recipe) == (rank, rank + oversample)
            # check that the core is identity
            @test sum(diag(recipe.core) .== 1) == rank
            @test size(recipe.core_view) == (2, 2)
            @test typeof(recipe.qr_decomp) <: SparseArrays.SPQR.QRSparse
        end

    end
    
    @testset "CrossApproximation: Update Core" begin
        # test with sparse matrix
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = sprand(n_rows, rank, .9) * sprand(rank, n_cols, .9)
            approx = CUR(
                rank, 
                oversample = oversample, 
                selector_cols = TestSelector()
            ) 
            core = CrossApproximation()

            # test the complete core function
            recipe = complete_core(approx, core, A)
            approx_recipe = complete_approximator(approx, A)
            # change indices selected to be first n
            approx_recipe.col_idx = 1:approx_recipe.n_col_vecs
            approx_recipe.row_idx = 1:approx_recipe.n_row_vecs
            # update the core recipe
            update_core!(recipe, approx_recipe, A)
            # check that a QR decomposition is stored of with correct sizes
            @test size(recipe) == (rank, rank + oversample)
            # check that the core is the correct QR decomposition
            @test typeof(recipe.qr_decomp) <: LinearAlgebra.QRCompactWY
            Q, R = qr(A[approx_recipe.row_idx, approx_recipe.col_idx])
            @test recipe.qr_decomp.Q == Q
            @test recipe.qr_decomp.R == R
        end

        # test with matrix
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = rand(n_rows, rank) * rand(rank, n_cols)
            approx = CUR(
                rank, 
                oversample = oversample, 
                selector_cols = TestSelector()
            ) 
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
            @test typeof(recipe.qr_decomp) <: SparseArrays.SPQR.QRSparse
            Q, R = qr(A[approx_recipe.row_idx, approx_recipe.col_idx])
            @test recipe.qr_decomp.Q == Q
            @test recipe.qr_decomp.R == R
        end

    end

    @testset "CrossApproximation: rapproximate" begin
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = rand(n_rows, rank) * rand(rank, n_cols)
            
            approx = CUR(
                rank, 
                oversample = oversample, 
                selector_cols = TestSelector()
            ) 
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

            approx = CUR(
                rank, 
                oversample = oversample, 
                selector_cols = TestSelector()
            ) 
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
            @test size(approx_recipe.U) == (col_idx, row_idx)
        end

    end

    @testset "CrossApproximation: mul!" begin
        # test multiplying the core matrix from the left
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = rand(n_rows, rank) * rand(rank, n_cols),
            B = ones(rank, rank + 1),
            C = ones(rank + oversample, rank + 1),
            x = ones(rank),
            y = ones(rank + oversample),
            alpha = 0.2,
            beta = 1.0

            approx = CUR(
                rank, 
                oversample = oversample, 
                selector_cols = TestSelector()
            ) 
            approx_recipe = complete_approximator(approx, A)
            approx_recipe.col_idx = 1:rank
            approx_recipe.row_idx = 1:rank + oversample
            core = CrossApproximation()
            recipe = complete_core(approx, core, A)
            update_core!(recipe, approx_recipe, A)
            # test for matrices
            Btest = deepcopy(B)
            mul!(B, recipe, C, alpha, beta)
            @test norm(B - ( beta * Btest + alpha * pinv(A[1:3, 1:2]) * C)) < 1e-10
            # test for vectors
            xtest = deepcopy(x)
            mul!(x, recipe, y, alpha, beta)
            @test norm(x - ( beta * xtest + alpha * pinv(A[1:3, 1:2]) * y)) < 1e-10
            # test for adjoints from the left
            Ctest = deepcopy(C)
            mul!(C, recipe', B, alpha, beta)
            @test norm(C - ( beta * Ctest + alpha * pinv(A[1:3, 1:2])' * B)) < 1e-10
            #test for adjoint with vector
            ytest = deepcopy(y)
            mul!(y, recipe', x, alpha, beta)
            @test norm(y - ( beta * ytest + alpha * pinv(A[1:3, 1:2])' * x)) < 1e-10
        end

        # test multiplying the core matrix from the right
        let n_rows = 10,
            n_cols = 10,
            rank = 2,
            oversample = 1,
            A = rand(n_rows, rank) * rand(rank, n_cols),
            B = ones(rank, rank + oversample),
            C = ones(rank + oversample, rank),
            x = ones(rank + oversample),
            y = ones(rank),
            alpha = 0.2,
            beta = 1.0

            approx = CUR(
                rank, 
                oversample = oversample, 
                selector_cols = TestSelector()
            ) 
            core = CrossApproximation()
            recipe = complete_core(approx, core, A) 
            # test for matrices
            Btest = deepcopy(B)
            mul!(B, C, recipe, alpha, beta)
            @test norm(B - ( beta * Btest + alpha * C * pinv(A[1:3, 1:2]))) < 1e-10
            # test for vectors
            xtest = deepcopy(x)
            mul!(x', y', recipe, alpha, beta)
            @test norm(x' - ( beta * xtest' + alpha * y' * pinv(A[1:3, 1:2]))) < 1e-10
            # test for adjoints from the left
            Ctest = deepcopy(C)
            mul!(C, B, recipe', alpha, beta)
            @test norm(C - ( beta * Ctest + alpha *  B * pinv(A[1:3, 1:2])')) < 1e-10
            #test for adjoint with vector
            ytest = deepcopy(y)
            mul!(y', x', recipe', alpha, beta)
            @test norm(y' - ( beta * ytest' + alpha * x' * pinv(A[1:3, 1:2])')) < 1e-10
        end

    end

end

end