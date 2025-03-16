module basic_logger
    using Test, RLinearAlgebra, Random
    include("../../test_helpers/field_test_macros.jl")
    include("../../test_helpers/approx_tol.jl")
    using .FieldTest
    using .ApproxTol
    @testset "Logger_BasicLogger" begin
        @test_logger BasicLoggerRecipe
    
        # Test default initlialization
        let
    	    Random.seed!(21321)
    	    L1 = BasicLogger()
    	    @test typeof(L1) <: Logger 
    	    @test L1.max_it == 0
    	    @test L1.collection_rate == 1
            @test L1.threshold_info == 0
    	    @test L1.stopping_criterion == threshold_stop
    	    # Test completion of constructor
            n_rows = 4
            n_cols = 2
    	    A = rand(n_rows, n_cols)
    	    b = rand(n_rows)
    	    L_method = complete_logger(L1, A, b)
    	    @test typeof(L_method) <: LoggerRecipe
            # test the initialization of logger
            @test L_method.max_it == 3 * n_rows
            @test size(L_method.hist, 1) == Int(ceil(3 * n_rows / 1)) + 1
            @test L_method.stopping_criterion == threshold_stop
            @test L_method.threshold_info == 0
            @test L_method.converged == false
            @test L_method.iteration == 1 
            @test L_method.record_location == 1
        end
            # Test initialization with keyword inputs
        let
    	    Random.seed!(21321)
            max_it = 1000
            cr = 2
            threshold = 1e-6
    	    L1 = BasicLogger(max_it = max_it,
                             collection_rate = cr,
                             threshold = threshold
                            )
    	    @test typeof(L1) <: Logger 
    	    @test L1.max_it == max_it 
    	    @test L1.collection_rate == cr
            @test L1.threshold_info == threshold
    	    @test L1.stopping_criterion == threshold_stop
    	    # Test completion of constructor
            n_rows = 4
            n_cols = 2
    	    A = rand(n_rows, n_cols)
    	    b = rand(n_rows)
    	    L_method = complete_logger(L1, A, b)
    	    @test typeof(L_method) <: LoggerRecipe
            # test the initialization of logger
            @test L_method.max_it == max_it 
            @test size(L_method.hist, 1) == Int(ceil(max_it / cr)) + 1
            @test L_method.stopping_criterion == threshold_stop
            @test L_method.threshold_info == threshold 
            @test L_method.converged == false
            @test L_method.iteration == 1 
            @test L_method.record_location == 1
        end
    
        # Test functionality of logger
        let 
    	    L1 = BasicLogger(collection_rate = 2, threshold = 1e-5)
            n_rows = 4
            n_cols = 2
    	    A = rand(n_rows, n_cols)
    	    b = rand(n_rows)
    	    L_method = complete_logger(L1, A, b)
            # first check that nothing gets recorded at it 1
            update_logger!(L_method, 1e-2, 1)
            @test L_method.hist[1] == 0.0 
            @test L_method.record_location == 1
            @test L_method.iteration == 1
            @test L_method.converged == false
            # Test if the threshold is satisfied on first it
            update_logger!(L_method, 1e-6, 1)
            @test L_method.hist[1] == 0.0 
            @test L_method.record_location == 1
            @test L_method.iteration == 1
            @test L_method.converged == true 
            # Check that a value is recorded at iteration 2
            update_logger!(L_method, 1e-2, 2)
            @test L_method.hist[1] == 1e-2 
            @test L_method.record_location == 2
            @test L_method.iteration == 2
            @test L_method.converged == false 
            # Check that a value is recorded at iteration 2 and record location reset when
            # threshold satisfied
            update_logger!(L_method, 1e-6, 2)
            @test L_method.hist[2] == 1e-6 
            @test L_method.record_location == 1
            @test L_method.iteration == 2
            @test L_method.converged == true 
            # check that you stop whe the max_it is satisfied max_it is 3 * num rows in A 
            update_logger!(L_method, 1e-1, 13)
            @test L_method.hist[1] == 1e-2 
            @test L_method.record_location == 1
            @test L_method.iteration == 13 
            @test L_method.converged == false 
            
    	end
    
    end

end
