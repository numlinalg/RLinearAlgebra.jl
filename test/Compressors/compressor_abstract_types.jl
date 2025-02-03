module compressor_abstract_types
    using Test, RLinearAlgebra
    include("../test_helpers/field_test_macros.jl")
    include("../test_helpers/approx_tol.jl")

    @test isdefined(Main, :Compressor)
    @test isdefined(Main, :CompressorRecipe)
    @test isdefined(Main, :CompressorAdjoint)

end
