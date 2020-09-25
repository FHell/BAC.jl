cd(@__DIR__)
using Pkg
Pkg.activate(".")
# using Revise

using Random
Random.seed!(42)

include("../src/BAC.jl")

ES = BAC.ExampleSystems
PL = BAC.PlotUtils

dim_sys = 10
N_samples = 10

bac_1 = ES.create_graph_example(dim_sys, 3, 0.:0.1:10., N_samples)

p_initial = ones(2*N_samples+dim_sys)

l, sol1, sol2 = bac_1(p_initial)

PL.plot_callback(p_initial, l, sol1, sol2)

bac_1.output_metric(sol1, sol2)

using DiffEqFlux

res = DiffEqFlux.sciml_train(
    bac_1,
    p_initial,
    DiffEqFlux.ADAM(0.1),
    maxiters = 5,
    cb = BAC.basic_bac_callback
    )

losses, sol1, sol2 = BAC.individual_losses(bac_1, res.minimizer);
PL.plot_callback(res.minimizer, sum(losses), sol1, sol2)

relu(x) = max(0., x) # We manually reset the minimizer to all positive as the function has a relu dependency on p

res = DiffEqFlux.sciml_train(
    bac_1,
    res.minimizer,
    DiffEqFlux.ADAM(0.001),
    maxiters = 10,
    cb = BAC.basic_bac_callback
    )

relu(x) = max(0., x) # We manually reset the minimizer to all positive as the function has a relu dependency on p

for i in 1:30
    global res
    res = DiffEqFlux.sciml_train(
        bac_1,
        relu.(res.minimizer),
        DiffEqFlux.BFGS(initial_stepnorm = 0.01),
        maxiters = 5,
        cb = BAC.basic_bac_callback
        )
    losses, sol1, sol2 = BAC.individual_losses(bac_1, res.minimizer);
    PL.plot_callback(res.minimizer, sum(losses), sol1, sol2)
end    
    