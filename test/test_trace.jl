using LinearAlgebra

A = rand(5, 5)
trace_la = tr(A)
trace_est = RLinearAlgebra.randomized_trace(A, 1000)

println("Trace: ", trace_la)
println("Trace approximation: ", trace_est)
println("Trace approximation error: ", trace_la - trace_est)
