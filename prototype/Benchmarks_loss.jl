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
using ParameterizedFunctions, DiffEqDevTools

include("../src/Core.jl")
include("../src/ExampleSystems.jl")
include("../src/PlotUtils.jl")
include("../src/Benchmark.jl")

Random.seed!(42);

dim_sys = 10

bac_10 = create_graph_example(dim_sys, 3, 0.:0.1:10., 10)

Random.seed!(42);
p_rands = [rand(30) .+ 0.5 for i in 1:20]
#length(p_rands)
abstols = 1.0 ./ 10.0 .^ (1:8)
reltols = 1.0 ./ 10.0 .^ (1:8)
bac_10.solver = Vern8()
test_ls = [bac_10(p; abstol=1e-14, reltol=1e-14) for p in p_rands]

setups = [Dict(:alg=>BS3()),
          Dict(:alg=>Tsit5()),
          Dict(:alg=>RK4()),
          Dict(:alg=>DP5()),
          Dict(:alg=>TRBDF2()),
#          Dict(:alg=>OwrenZen3()),
#          Dict(:alg=>OwrenZen4()),
          Dict(:alg=>OwrenZen5())]

lp = LossPrecisionSet(p_rands, abstols, reltols, setups;appxsol=test_ls)
plot(lp)
