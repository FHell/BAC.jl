# For exploring features in the package here is a non package include based environment for running the code.
cd(@__DIR__)
using Pkg
Pkg.activate(".")
# Pkg.instantiate()

using Random
using DiffEqFlux
using OrdinaryDiffEq
using Plots
using LightGraphs
using Statistics

include("../src/Core.jl")
include("../src/ExampleSystems.jl")
include("../src/PlotUtils.jl")

Random.seed!(42);

dim_sys = 10

bac_10 = create_graph_example(dim_sys, 3, 0.:0.1:10., 10)

# a nonlinear diffusively coupled graph system.
# Specification is a two node version of the graph.
# => Can we make a large graph react like a small graph by tuning the non-linearity at the nodes?

p_initial = ones(2*10+dim_sys)

# bac_1 implements the loss function. We are looking for parameters that minimize it, it can be evaluated
# directly on a parameter array:
l = bac_10(p_initial) # 110
l = bac_10(p_initial; abstol=1e-2, reltol=1e-2) # 108

p_rands = [rand(30) .+ 0.5 for i in 1:20]
@time ls_2 = [bac_10(p; abstol=1e-2, reltol=1e-2) for p in p_rands]
@time ls_3 = [bac_10(p; abstol=1e-3, reltol=1e-3) for p in p_rands]
@time ls_4 = [bac_10(p; abstol=1e-4, reltol=1e-4) for p in p_rands]
@time ls_5 = [bac_10(p; abstol=1e-5, reltol=1e-5) for p in p_rands]
@time ls_6 = [bac_10(p; abstol=1e-6, reltol=1e-6) for p in p_rands]

bac_10.solver=TRBDF2()
@time sls_2 = [bac_10(p; abstol=1e-2, reltol=1e-2) for p in p_rands]
@time sls_3 = [bac_10(p; abstol=1e-3, reltol=1e-3) for p in p_rands]
@time sls_4 = [bac_10(p; abstol=1e-4, reltol=1e-4) for p in p_rands]
@time sls_5 = [bac_10(p; abstol=1e-5, reltol=1e-5) for p in p_rands]
@time sls_6 = [bac_10(p; abstol=1e-10, reltol=1e-10) for p in p_rands]

println(mean(sls_5 .- sls_6))
println(mean(ls_6 .- sls_6))
println(mean(ls_5 .- sls_6))

using Zygote
gradient(p -> bac_10(p; abstol=1e-3, reltol=1e-3), p_rands[1])
@time gradient(p -> bac_10(p; abstol=1e-3, reltol=1e-3), p_rands[1])
