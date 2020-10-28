## In VSCode double hashes demarcate code cells that can be run with Shift-Enter

# For exploring features in the package here is a non package include based environment for running the code.
cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Random
using DiffEqFlux
using OrdinaryDiffEq
using Plots
using LightGraphs
using Statistics
using DataFrames, Pipe

include("../src/Core.jl")
include("../src/ExampleSystems.jl")
include("../src/PlotUtils.jl")
include("../src/Benchmark.jl")

Random.seed!(42);

dim_sys = 10

bac_10 = create_graph_example(dim_sys, 3, 0.:0.1:10., 10)

# we can easily plot the input sample
plot(0:0.01:10, bac_10.input_sample, c=:gray, alpha=1, legend=false)

# a nonlinear diffusively coupled graph system.
# Specification is a two node version of the graph.
# => Can we make a large graph react like a small graph by tuning the non-linearity at the nodes?

p_initial = ones(2*10+dim_sys)

## Finished initialization
## Benchmarking

# Implement training with a set of optimizers
setups = [Dict(:opt=>DiffEqFlux.ADAM(0.1), :name=>"ADAM(0.1)"),
          Dict(:opt=>DiffEqFlux.Descent(0.1), :name=>"Descent(0.1)"),
          Dict(:opt=>DiffEqFlux.AMSGrad(0.1), :name=>"AMSGrad(0.1)"),
          Dict(:opt=>DiffEqFlux.NelderMead(), :name=>"NelderMead()"),
          Dict(:opt=>DiffEqFlux.BFGS(initial_stepnorm = 0.01), :name=>"BFGS(initial_stepnorm = 0.01)"),
          # Dict(:opt=>DiffEqFlux.BFGS(initial_stepnorm = 0.1), :name=>"BFGS(initial_stepnorm = 0.1)"), #seems to get stuck
          Dict(:opt=>DiffEqFlux.MomentumGradientDescent(), :name=>"MomentumGradientDescent()")]

t, l = partySet(1, setups, bac_10, p_initial) #Compiling everything
t, l = partySet(20, setups, bac_10, p_initial)

#tN, lN = party_new(3, DiffEqFlux.NewtonTrustRegion(), bac_100, relu.(res_100.minimizer))
#Above with error message TypeError: in typeassert, expected Float64, got a value of type ForwardDiff.Dual{Nothing,Float64,12}

# BFGS seems to get stuck with too larg maxiter or stepnorm
# tB, lB = party_new(20, DiffEqFlux.BFGS(initial_stepnorm = 0.01), bac_10, p_initial)

# Write training data into DataFrame
begin
    BenchResults = DataFrame(solver = String[], times = Float64[], loss = Float64[])
    for i in 1:length(setups)
        for j in 1:length(t[i])
            #BenchResults.solver = repeat(setups[i][:name], length(t[i]))
            push!(BenchResults.solver, setups[i][:name])
            push!(BenchResults.times, t[i][j])
            push!(BenchResults.loss, l[i][j])
        end
    end
    display(BenchResults)
end

# Display of DataFrame grouped with different solver
for i in 1:length(setups)
    display(BenchResults[BenchResults.solver.==setups[i][:name],:])
end

# Plot of time-loss figure
begin
    tempt = filter(:solver => ==(setups[1][:name]), BenchResults)
    plt = scatter(tempt.times, tempt.loss,
            title = "Training loss value",
            label = setups[1][:name])
    for i in 2:length(setups)
        tempt = filter(:solver => ==(setups[i][:name]), BenchResults)
        plt = scatter!(tempt.times, tempt.loss,
                label = setups[i][:name])

    end
    display(plt)
end

# StatsPlots support DataFrames with a marco @df for easy ploting in different solver.
using StatsPlots
@df BenchResults scatter(
    :times,
    :loss,
    group = :solver,
    title = "Training loss value",
    m = (0.8, [:+ :h :star7], 7),
    bg = RGB(0.2, 0.2, 0.2)
)
##