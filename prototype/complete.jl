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
using LaTeXStrings

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

# bac_1 implements the loss function. We are looking for parameters that minimize it, it can be evaluated
# directly on a parameter array:
l = bac_10(p_initial) # 1713
l = bac_10(p_initial; abstol=1e-2, reltol=1e-2) # 1868

# Plot callback plots the solutions passed to it:
samples = 1:3 # we can choose what samples to use for plots everywhere
plot_callback(bac_10, p_initial, l, input_sample=samples, fig_name = "../graphics/test/initial123_l$l.png")

#losses = zeros(0) # for plotting loss change over the course of optimization
# Underlying the loss function is the output metric comparing the two trajectories:
sol1, sol2 = solve_bl_n(bac_10, 3, p_initial, input_sample=samples)
bac_10.output_metric(sol1, sol2)

# We can get all the individual contributions with
il = individual_losses(bac_10, p_initial)

# Train with 10 samples, low accuracy and relatively large ADAM step size: (1.5 minutes on my Laptop)
@time res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p; abstol=1e-2, reltol=1e-2),
    p_initial,
    DiffEqFlux.ADAM(0.5),
    maxiters = 5,
    #cb = basic_bac_callback
    cb = (p, l) -> plot_callback(bac_10, p, l, input_sample=samples, fig_name = "../graphics/test/res10_$(l).png")
    )

plot_callback(bac_10, res_10.minimizer, l, input_sample = 1:3)

# Train with 10 samples, medium accuracy and still large ADAM step size:
@time res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p; abstol=1e-6, reltol=1e-6),
    res_10.minimizer,
    DiffEqFlux.ADAM(0.5),
    maxiters = 20,
    #cb = basic_bac_callback
    cb = (p, l) -> plot_callback(bac_10, p, l, input_sample = samples, fig_name = "../graphics/test/res10_$(l).png")
    )

@time res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p),
    res_10.minimizer,
    DiffEqFlux.ADAM(0.5),
    maxiters = 20,
    cb = basic_bac_callback
    # cb = (p, l) -> plot_callback(bac_10, p, l, input_sample=samples)
    )

plot_callback(bac_10, res_10.minimizer, l; input_sample=samples, fig_name = "../graphics/test_opt_123-10_l$l.png")
# this got it down to 0.01 loss. we can look at the minimizer:
# p_sys
res_10.minimizer[1:dim_sys] |> relu |> println
# p_spec
res_10.minimizer[dim_sys+1:end] |> relu |> println

# After a few runs of the above code block (trying both larger and smaller ADAM step sizes) I get this down to < 0.3
# At this point I don't really care about going further as we are probably overfitting to the small sample.

# We can check the quality of the resulting minimizer by optimizing the specs only (a much simpler repeated 2d optimization problem)
p2 = bac_spec_only(bac_10, res_10.minimizer)
losses = individual_losses(bac_10, p2)
median(losses) #0.02

# In order to understand how much we were overfitting with respect to the concrete sample, we resample
# That is, we generate a new problem with a new sample from the same distribution:
bac_10_rs = resample(rand_fourier_input_generator, bac_10)
p3 = bac_spec_only(bac_10_rs, res_10.minimizer)
losses_rs = individual_losses(bac_10_rs, p3)
median(losses_rs) #0.04
# This will give us information on the system tuning with a sample different from the one that the tuning was optimized for.

## Larger number of samples

# We warmed up the optimization with a very small number of samples,
# we can now initialize a higher sampled optimization using the system parameters found in the lower one:

p_100 = ones(2*100+dim_sys)
p_100[1:dim_sys] .= res_10.minimizer[1:dim_sys]
p_100[dim_sys+1:end] .= repeat(res_10.minimizer[dim_sys+1:dim_sys+2], 100)

bac_100 = resample(rand_fourier_input_generator, bac_10; n = 100)

# Optimizing only the specs is a task linear in the number of samples,
# the idea is that this will help with warming up the optimization
# We can also study the quality of the tuning found by the optimization based on a small number of Samples
p_100_initial = bac_spec_only(bac_100, p_100;
                    optimizer=DiffEqFlux.ADAM(0.1),
                    optimizer_options=(:maxiters => 10,),
                    abstol = 1e-3, reltol=1e-3)
losses_100_initial = individual_losses(bac_100, p_100_initial)
# Todo: Indication of bug or not completely understood behaviour!!
median(losses_100_initial) # This is much larger (factor 5-10) than the losses_10_rs version. It shouldn't be. Needs to be investigated!!!!!
# Possibility: THe optimization in bac_spec_only is not doing its job very well, switch to ADAM?
plot_callback(bac_100, p_100_initial, l, input_sample = samples)

# Train the full system:
@time res_100 = DiffEqFlux.sciml_train(
    bac_100,
    p_100_initial,
    # DiffEqFlux.ADAM(0.5),
    DiffEqFlux.BFGS(initial_stepnorm = 0.01),
    maxiters = 5,
    cb = basic_bac_callback
    )

# Continue improving it for 50 Steps with some plotting in between:
for i in 1:10
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
    plot_callback(bac_100, res_100.minimizer, l, input_sample = samples, fig_name = "../graphics/test_res_100_int"*string(i, pad = 2)#=;ylims = (-0.5,0.5)=#)
end

#plot_callback(bac_10, res_10.minimizer, l, input_sample = samples, legend=false, fig_name = "../graphics/for Frank/opt_1-3_nolegend.png")

#plot_callback(bac_10, p_initial, l, input_sample = 1:10, legend = false, fig_name = "../graphics/for Frank/init_1-10_nolegend.png")
#plot_callback(bac_10, res_10.minimizer, l, input_sample = 1:10, legend=false, fig_name = "../graphics/for Frank/opt_1-10_nolegend.png")
#plot_callback(bac_100, res_100.minimizer, l, input_sample = 50:59)
## Benchmarking

# Implement training with a set of optimizers
setups = [Dict(:opt=>DiffEqFlux.ADAM(0.1), :name=>"ADAM(0.1)"),
          Dict(:opt=>DiffEqFlux.Descent(0.1), :name=>"Descent(0.1)"),
          Dict(:opt=>DiffEqFlux.AMSGrad(0.1), :name=>"AMSGrad(0.1)"),
          Dict(:opt=>DiffEqFlux.NelderMead(), :name=>"NelderMead()")]
#          Dict(:opt=>DiffEqFlux.BFGS(initial_stepnorm = 0.1), :name=>"BFGS(initial_stepnorm = 0.01)")]
t, l = train_set(5, setups, bac_10, p_initial)
#tN, lN = single_train(3, DiffEqFlux.NewtonTrustRegion(), bac_100, relu.(res_100.minimizer))
#Above with error message TypeError: in typeassert, expected Float64, got a value of type ForwardDiff.Dual{Nothing,Float64,12}
tM, lM = single_train(5, DiffEqFlux.MomentumGradientDescent(), bac_10, p_initial)
tB, lB = single_train(5, DiffEqFlux.BFGS(initial_stepnorm = 0.1), bac_10, p_initial)

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
using LaTeXStrings
losses = individual_losses(bac_100, res_100.minimizer)
confidence_interval(losses, 0.3)
x= exp10.(range(log10(.01),stop=log10(0.2), length = 50))
plot(x, (x)->1-confidence_interval(losses, x)[1],
    xlabel = "ε", ylabel=L"\hat d^{\rho,\varepsilon}",legend=:bottomright, label=false,c=:blue)
    #label="Fraction of samples within set distance from specification")
    plot!(x, (x)->1-confidence_interval(losses, x)[2],
        xlabel = "ε", ylabel=L"\hat d^{\rho,\varepsilon}",legend=:bottomright, label=false, linestyle=:dash, c=:blue)
        #label="Fraction of samples within set distance from specification")
    plot!(x, (x)->1-confidence_interval(losses, x)[3],
        xlabel = "ε", ylabel=L"\hat d^{\rho,\varepsilon}",legend=:bottomright, label=false, linestyle=:dash, c=:blue)
        #label="Fraction of samples within set distance from specification")
savefig("../graphics/confidence_int_nolabel_new.png")

plot(sort!(losses),[1:length(losses)]./length(losses), label = false)
