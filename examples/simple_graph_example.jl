using BAC

using Random
using Plots
Random.seed!(42);
using GraphPlot
using DiffEqFlux
using Pipe: @pipe
using Statistics
using LaTeXStrings

#=
## Simple graph example
As an example we want to tune a system of 10 nodes with one grid connection point to a specification of 2 nodes.

Our goal is to find such parameters for the 10-node system that it behaves close enough to a 2-node system under all possible inputs.
In this example we start with just 10 possible inputs.
=#
dim_sys = 10
bac_10 = BAC.create_graph_example(dim_sys, 3, 0.:0.1:10., 10);
#=
We can plot the input trajectories and system layouts. Optimized system is a nonlinear diffusively coupled graph system.
Specification is a two node version of the graph.
=#
plot(0:0.01:10, bac_10.scenario_nums, c=:gray, alpha=1, legend=false)
BAC.plot_sys_graph(bac_10)
# => Can we make a large graph react like a small graph by tuning the non-linearity at the nodes?

p_initial = ones(2*10+dim_sys)
#=
## Finished initialization

bac_10 implements the loss function. We are looking for parameters that minimize it. It can be evaluated directly on a parameter array:
=#
l = bac_10(p_initial; abstol=1e-2, reltol=1e-2)

#=
Plot callback plots the solutions passed to it.
We choose which scenarios to use for plots, as plotting for all scenarios makes visual results harder to understand.
=#
scenarios = 1:3 # we can choose what scnearios to use for plots everywhere
BAC.plot_callback(bac_10, p_initial, l, scenario_nums=scenarios, fig_name = "graphics/initial10_$(l).png")

#losses = zeros(0) # for plotting loss change over the course of optimization
#=
Underlying the loss function is the output metric comparing the two trajectories:
=#
sol1, sol2 = solve_bl_n(bac_10, 3, p_initial, scenario_nums=scenarios)
bac_10.output_metric(sol1, sol2)

#= We can get all the individual contributions with individual_losses:=#
il = individual_losses(bac_10, p_initial)

#= Training with 10 scenarios, low accuracy and relatively large ADAM step size is a way to quickly get a good approximation, that can then be improved by increasing accuracy and number of samples.
=#
@time res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p; abstol=1e-2, reltol=1e-2),
    p_initial,
    DiffEqFlux.ADAM(0.5),
    maxiters = 5,
    cb = (p, l) -> plot_callback(bac_10, p, l, scenario_nums=scenarios)
    )

plot_callback(bac_10, res_10.minimizer, l, scenario_nums = 1:3)

@time res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p; abstol=1e-6, reltol=1e-6),
    res_10.minimizer,
    DiffEqFlux.ADAM(0.5),
    maxiters = 20,
    cb = (p, l) -> plot_callback(bac_10, p, l, scenario_nums = scenarios)
    )

@time res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p),
    res_10.minimizer,
    DiffEqFlux.ADAM(0.5),
    maxiters = 20,
    cb = basic_bac_callback
    )

plot_callback(bac_10, res_10.minimizer, l; scenario_nums=scenarios)
#=
After getting a good initial approximation, we can look at the minimizer.
Going further makes little sense as we would be overfitting to the small sample.
=#
res_10.minimizer[1:dim_sys] |> BAC.relu |> println #p_sys
res_10.minimizer[dim_sys+1:end] |> BAC.relu |> println #p_spec

#= ## Resampling
We can check the quality of the resulting minimizer by optimizing the specs only (a much simpler repeated 2d optimization problem)
=#
p2 = bac_spec_only(bac_10, res_10.minimizer)
losses = individual_losses(bac_10, p2)
median(losses)
#=
In order to understand how much we were overfitting with respect to the concrete sample, we resample.
That is, we generate a new problem with a new sample from the same distribution:
This will give us information on the system tuning with a sample different from the one that the tuning was optimized for.
=#
bac_10_rs = resample(BAC.rand_fourier_input_generator, bac_10);
p3 = bac_spec_only(bac_10_rs, res_10.minimizer)
losses_rs = individual_losses(bac_10_rs, p3)
median(losses_rs)

#=
The median individual loss has gone up by a factor of 4. This means that the system is somewhat overfit to the initial sample.

## Larger number of samples

We warmed up the optimization with a very small number of scenarios, we can now initialize a higher sampled optimization using the system parameters found in the lower one:
=#
p_100 = ones(2*100+dim_sys)
p_100[1:dim_sys] .= res_10.minimizer[1:dim_sys]
p_100[dim_sys+1:end] .= repeat(res_10.minimizer[dim_sys+1:dim_sys+2], 100)

bac_100 = resample(BAC.rand_fourier_input_generator, bac_10; n = 100);

#=
Optimizing only the specs is a task linear in the number of scenarios, the idea is that this will help with warming up the optimization.
We can also study the quality of the tuning found by the optimization based on a small number of samples.
=#
p_100_initial = bac_spec_only(bac_100, p_100;
                    optimizer=DiffEqFlux.ADAM(0.1),
                    optimizer_options=(:maxiters => 10,),
                    abstol = 1e-3, reltol=1e-3)
losses_100_initial = individual_losses(bac_100, p_100_initial)
median(losses_100_initial)
plot_callback(bac_100, p_100_initial, l, scenario_nums = scenarios)

#= Now we can train the full system:
=#
@time   res_100 = DiffEqFlux.sciml_train(
    bac_100,
    p_100_initial,
    # DiffEqFlux.ADAM(0.5),
    DiffEqFlux.BFGS(initial_stepnorm = 0.01),
    maxiters = 5,
    cb = basic_bac_callback
    )

#= Continue improving it for 150 Steps with some plotting in between:=#
for i in 1:30
    global res_100
    res_100 = DiffEqFlux.sciml_train(
        bac_100,
        relu.(res_100.minimizer),
        DiffEqFlux.ADAM(0.1),
        # DiffEqFlux.BFGS(initial_stepnorm = 0.01),
        maxiters = 5,
        cb = basic_bac_callback
        )
    l = bac_100(res_100.minimizer);
    plot_callback(bac_100, res_100.minimizer, l, scenario_nums = scenarios)
end

plot_callback(bac_10, res_10.minimizer, l; scenario_nums=scenarios)

losses = individual_losses(bac_100, res_100.minimizer)

x= exp10.(range(log10(.05),stop=log10(0.4), length = 50))
plot(x, (x)->1-confidence_interval(losses, x)[1],
    xlabel = "ε", ylabel=L"\hat d^{\rho,\varepsilon}",legend=:bottomright, label=false,c=:blue)
    #label="Fraction of scenarios within set distance from specification")
    plot!(x, (x)->1-confidence_interval(losses, x)[2],
        xlabel = "ε", ylabel=L"\hat d^{\rho,\varepsilon}",legend=:bottomright, label=false, linestyle=:dash, c=:blue)
        #label="Fraction of scenarios within set distance from specification")
    plot!(x, (x)->1-confidence_interval(losses, x)[3],
        xlabel = "ε", ylabel=L"\hat d^{\rho,\varepsilon}",legend=:bottomright, label=false, linestyle=:dash, c=:blue)
        #label="Fraction of scenarios within set distance from specification")

plot(sort!(losses),[1:length(losses)]./length(losses), label = false)

plot(sort!(losses),[1:length(losses)]./length(losses), label = false)
