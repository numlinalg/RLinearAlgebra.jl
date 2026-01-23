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

# Test core structures
mutable struct TestCore <: CURCore end
mutable struct TestCoreRecipe <: CURCoreRecipe
    A::AbstractMatrix
end

RLinearAlgebra.complete_core(core::CURCore, cur::CUR, A::AbstractMatrix)
    return TestCoreRecipe(rand(2,2))
end

RLinearAlgebra.update_core!(core::CURCoreRecipe, cur::CURRecipe, A::AbstractMatrix)
    core.A = pinv(A[cur.row_idx, cur.col_idx])
    return nothing
end

@testset "CUR" begin
    @testset "CUR" begin
        supertype(CUR) == Approximator

        # test the fieldnames and types
        fieldnames(CUR) == (
            :rank, 
            :oversample, 
            :col_selector, 
            :row_selector, 
            :core, 
            :block_size
        )
        fieldtypes(CUR) == (
            Int64,
            Int64,
            Selector,
            Selector,
            CURCore,
            Int64
        )

        # test errors
        let block_size = 1,
            rank = -1,
            oversample = 1,

            @test_throws CUR(
                rank = rank, 
                block_size = block_size, 
                oversample = oversample
            ) ArgumentError(
                "Field `rank` must be non-negative."
            )
        end
        
        let block_size = 1,
            rank = 1,
            oversample = -1,

            @test_throws CUR(
                rank = rank, 
                block_size = block_size, 
                oversample = oversample
            ) ArgumentError(
                "Field `oversample` must be non-negative."
            )
        end

        let block_size = -1,
            rank = 1,
            oversample = 1,

            @test_throws CUR(
                rank = rank, 
                block_size = block_size, 
                oversample = oversample
            ) ArgumentError(
                "Field `block_size` must be non-negative."
            )
        end

        # Test Constructor
        let selector = TestSelector(),
            block_size = 1,
            rank = 1,
            oversample = 1, 
            core = TestCore

            cur = CUR(
                rank = rank,
                oversample = oversample,
                core = core,
                block_size = block_size,
                col_selector = TestSelector()
            )

            @test typeof(cur.core) == TestCore
            @test typeof(cur.col_selector) == TestSelector
            @test typeof(cur.row_selector) == TestSelector
            @test cur.oversample == oversample
            @test cur.block_size === block_size
            @test cur.rank == rank
        end

    end 

    @testset "CURRecipe" begin
        supertype(CURRecipe) == ApproximatorRecipe

        # test the fieldnames and types
        fieldnames(CURRecipe) == (
            :n_row_vecs,
            :n_col_vecs,
            :col_selector,
            :row_selector,
            :row_idx,
            :col_idx,
            :C,
            :U,
            :R,
            :buffer_row,
            :buffer_core
        )
        fieldtypes(CURRecipe) == (
            Int64,
            Int64,
            SelectorRecipe,
            SelectorRecipe,
            Vector{Int64},
            Vector{Int64},
            AbstractMatrix,
            CURCoreRecipe,
            AbstractMatrix,
            AbstractArray,
            AbastractArray
        )
    end 

    @testset "CUR: Complete Approximator" begin

    end


    @testset "CUR: mul!" begin

    end

end

end