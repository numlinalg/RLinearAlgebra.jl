module approximators_abstract_types
    using Test, RLinearAlgebra
    include("../test_helpers/field_test_macros.jl")
    include("../test_helpers/approx_tol.jl")

    @testset "Approximator Abstract Types" begin
        @test isdefined(Main, :Approximator)
        @test isdefined(Main, :ApproximatorRecipe)
        @test isdefined(Main, :ApproximatorError)
        @test isdefined(Main, :ApproximatorErrorRecipe)
        @test isdefined(Main, :ApproximatorAdjoint)
    end

end
