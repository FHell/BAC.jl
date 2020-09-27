cd(@__DIR__)
using Pkg
Pkg.activate(".")
# using Revise

using Random
using DiffEqFlux
using OrdinaryDiffEq

Random.seed!(42);

include("../src/BAC.jl")

ES = BAC.ExampleSystems
PL = BAC.PlotUtils

dim_sys = 10

bac_10 = ES.create_graph_example(dim_sys, 3, 0.:0.1:10., 10)

bac_100 = ES.create_graph_example(dim_sys, 3, 0.:0.1:10., 100)
# a nonlinear diffusively coupled graph system.
# Specification is a two node version of the graph.
# => Can we make a large graph react like a small graph by tuning the non-linearity at the nodes?

p_initial = ones(2*10+dim_sys)

# bac_1 implements the loss function. We are looking for parameters that minimize it, it can be evaluated
# directly on a parameter array:
l, sol1, sol2 = bac_10(p_initial)

# Plot callback plots the solutions passed to it:
PL.plot_callback(p_initial, l, sol1, sol2)

# Underlying the loss function is the output metric comparing the two trajectories:
bac_10.output_metric(sol1, sol2)

# Test that training works:
res = DiffEqFlux.sciml_train(
    bac_10,
    p_initial,
    DiffEqFlux.ADAM(0.1),
    maxiters = 5,
    cb = BAC.basic_bac_callback
    )

p2 = bac_spec_only(bac_10, res.minimizer)


losses = BAC.individual_losses(bac_1, res.minimizer)

relu(x) = max(0., x) # We manually reset the minimizer to all positive as the function has a relu dependency on p

res = DiffEqFlux.sciml_train(
    bac_1,
    relu.(res.minimizer),
    DiffEqFlux.ADAM(0.05),
    maxiters = 10,
    cb = BAC.basic_bac_callback
    )

for i in 1:30
    global res
    res = DiffEqFlux.sciml_train(
        bac_1,
        relu.(res.minimizer),
        DiffEqFlux.ADAM(0.05),
        # DiffEqFlux.BFGS(initial_stepnorm = 0.01),
        maxiters = 5,
        cb = BAC.basic_bac_callback
        )
    l, sol1, sol2 = bac_1(res.minimizer);
    PL.plot_callback(res.minimizer, l, sol1, sol2)
end    
    