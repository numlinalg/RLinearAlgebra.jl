module distribution_abstract_types
using Test, RLinearAlgebra
import LinearAlgebra: ldiv!
using ..FieldTest
using ..ApproxTol
struct TestDistribution <: Distribution end
struct TestDistributionRecipe <: DistributionRecipe end

@testset "Distribution Abstract Types" begin
    @test isdefined(Main, :Distribution)
    @test isdefined(Main, :DistributionRecipe)
end

@testset "Distribution Argument Errors" begin
    A = rand(2, 2)
    b = rand(2)
    x = rand(2)

    @test_throws ArgumentError complete_distribution(TestDistribution(), A)
    @test_throws ArgumentError update_distribution!(TestDistributionRecipe(), A)
end

end