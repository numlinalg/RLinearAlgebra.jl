module logger_abstract_types
    using Test, RLinearAlgebra
    include("../../test_helpers/field_test_macros.jl")
    include("../../test_helpers/approx_tol.jl")
    
    @testset "Logger Abstract Types" begin
        @test isdefined(Main, :Logger)
        @test isdefined(Main, :LoggerRecipe)
    end

end
