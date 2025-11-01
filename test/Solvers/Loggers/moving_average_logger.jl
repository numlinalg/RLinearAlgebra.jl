module moving_average_logger
    using Test, RLinearAlgebra, Random
    include("../../test_helpers/field_test_macros.jl")
    include("../../test_helpers/approx_tol.jl")
    using .FieldTest
    using .ApproxTol
    @testset "Logger MALogger" begin
        Random.seed!(21321)
        n_rows = 4
        n_cols = 2
        A = rand(n_rows, n_cols)
        b = rand(n_rows)

        # How to use the logger, the error for the update
        a = MALogger()
        b = complete_logger(a)
        reset_logger!(b)
        update_logger!(b, 0.5, 1)


        



    end

end
