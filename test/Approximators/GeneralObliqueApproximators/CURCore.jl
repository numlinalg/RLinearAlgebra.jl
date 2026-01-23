module CURCore 
using Test, RLinearAlgebra, LinearAlgebra

struct TestCURCore <: CURCore end 
struct TestCURCoreRecipe <: CURCoreRecipe 
    code::Int64
end 

struct TestSelectorRecipe <: SelectorRecipe end

@testset "CURCore Abstract Types" begin
    @test isdefined(Main, :CURCore)
    @test isdefined(Main, :CURCoreRecipe)
end

@testset "CURCore Completion Errors" begin
    cur = CUR(1)
    A = ones(2, 2)
    core = TestCURCore()
    
    @test_throws ArgumentError complete_core(core, cur, A)
end

@testset "CURCore Update Errors" begin
    cur = CURRecipe{TestCURCoreRecipe}(
        2, 
        2,
        TestSelectorRecipe(),
        TestSelectorRecipe(),
        ones(Int64, 2),
        ones(Int64, 2),
        zeros(2, 2), 
        TestCURCoreRecipe(1),
        zeros(2, 2),
        zeros(1), 
        zeros(1)
    )
    A = ones(2, 2)
    @test_throws ArgumentError update_core!(cur.U, cur, A)
end

end