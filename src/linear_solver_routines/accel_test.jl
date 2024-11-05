using Revise
using Random
using LinearAlgebra
using MatrixDepot
using Plots
using RLinearAlgebra
# seed 1232, iter = 100000, alpha = 1.0015625, beta = 0.018, size = 20
Random.seed!(123132)
rows = 128
cols = 128 
iter = 100 
thres = 1e-36
b_size = 20 

A = randn(rows, cols)#"phillips"
#A = matrixdepot("chebspec", rows)
A = matrixdepot("cauchy", rows)
x = randn(cols)
b = A * x
s1 = svd(A).S[1]
sn = svd(A).S[end]
solver1 = RLSSolver(
                    LinSysBlkColReplace(block_size = b_size),
                    LinSysBlkColGent(), 
                    LSLogMA(), 
                    LSStopThreshold(iter, thres), 
                    nothing
                   )

betao = ((s1 - sn) / (s1 + sn))^2
alphao = 4 / (s1 + sn)^2
solver2 = RLSSolver(
                    LinSysBlkColReplace(block_size = b_size),
                    LinSysBlkColGent(), 
                    LSLogMA(), 
                    LSStopThreshold(iter, thres), 
                    nothing
                   )
#=solver2 = RLSSolver(
                    LinSysBlkColReplace(block_size = b_size),
                    LinSysBlkColGentAccel(
                                          α = 0.1,
                                          beta = 0,
                                          maxit = iter,
                                        ), 
                    LSLogMA(), 
                    LSStopThreshold(iter, thres), 
                    nothing
                   )
=#
H = hadamard(cols) ./ sqrt(cols)
Random.seed!(123)
t1 = @elapsed x1 = rsolve(solver1, A, b)
Random.seed!(123)
t2 = @elapsed x2 = rsolve(solver2, A * H, b)

plot(solver1.log.resid_hist[solver1.log.resid_hist .> 1e-32], 
     lab = "CD", 
     yscale = :log10,
     title = "Max ev $s1"
    )
    
plot!(solver2.log.resid_hist[solver2.log.resid_hist .> 1e-32], 
      lab = "Accel")
