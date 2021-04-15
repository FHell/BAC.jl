using BAC

using Random
using Plots
Random.seed!(42);
using GraphPlot
using DiffEqFlux

dim_sys = 10

bac_10 = BAC.create_graph_example(dim_sys, 3, 0.:0.1:10., 10)

plot(0:0.01:10, bac_10.input_sample, c=:gray, alpha=1, legend=false)
BAC.plot_sys_graph(bac_10)

p_initial = ones(2*10+dim_sys)

# Finished initialization

l = bac_10(p_initial) # 1713
l = bac_10(p_initial; abstol=1e-2, reltol=1e-2) # 1868

samples = 1:3 # we can choose what samples to use for plots everywhere
BAC.plot_callback(bac_10, p_initial, l, input_sample=samples)

losses = zeros(0) # for plotting loss change over the course of optimization

sol1, sol2 = solve_bl_n(bac_10, 3, p_initial, input_sample=samples)
bac_10.output_metric(sol1, sol2)

il = individual_losses(bac_10, p_initial)

@time res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p; abstol=1e-2, reltol=1e-2),
    p_initial,
    DiffEqFlux.ADAM(0.5),
    maxiters = 5,
    #cb = basic_bac_callback
    cb = (p, l) -> plot_callback(bac_10, p, l, input_sample=samples)
    )

plot_callback(bac_10, res_10.minimizer, l, input_sample = 1:3)

@time res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p; abstol=1e-6, reltol=1e-6),
    res_10.minimizer,
    DiffEqFlux.ADAM(0.5),
    maxiters = 20,
    #cb = basic_bac_callback
    cb = (p, l) -> plot_callback(bac_10, p, l, input_sample = samples)
    )

@time res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p),
    res_10.minimizer,
    DiffEqFlux.ADAM(0.5),
    maxiters = 20,
    cb = basic_bac_callback

    )

plot_callback(bac_10, res_10.minimizer, l; input_sample=samples, fig_name = "../graphics/opt_123-10_l$(l)_axis.png")

res_10.minimizer[1:dim_sys] |> relu |> println

res_10.minimizer[dim_sys+1:end] |> relu |> println

p2 = bac_spec_only(bac_10, res_10.minimizer)
losses = individual_losses(bac_10, p2)
median(losses) #0.02

bac_10_rs = resample(rand_fourier_input_generator, bac_10)
p3 = bac_spec_only(bac_10_rs, res_10.minimizer)
losses_rs = individual_losses(bac_10_rs, p3)
median(losses_rs) #0.04

# Larger number of samples

p_100 = ones(2*100+dim_sys)
p_100[1:dim_sys] .= res_10.minimizer[1:dim_sys]
p_100[dim_sys+1:end] .= repeat(res_10.minimizer[dim_sys+1:dim_sys+2], 100)

bac_100 = resample(rand_fourier_input_generator, bac_10; n = 100)

p_100_initial = bac_spec_only(bac_100, p_100;
                    optimizer=DiffEqFlux.ADAM(0.1),
                    optimizer_options=(:maxiters => 10,),
                    abstol = 1e-3, reltol=1e-3)
losses_100_initial = individual_losses(bac_100, p_100_initial)

median(losses_100_initial) # This is much larger (factor 5-10) than the losses_10_rs version. It shouldn't be. Needs to be investigated!!!!!

plot_callback(bac_100, p_100_initial, l, input_sample = samples)

@time   res_100 = DiffEqFlux.sciml_train(
    bac_100,
    p_100_initial,

    DiffEqFlux.BFGS(initial_stepnorm = 0.01),
    maxiters = 5,
    cb = basic_bac_callback
    )

for i in 1:10
    global res_100
    res_100 = DiffEqFlux.sciml_train(
        bac_100,
        relu.(res_100.minimizer),
        DiffEqFlux.ADAM(0.1),

        maxiters = 5,
        cb = basic_bac_callback
        )
    l = bac_100(res_100.minimizer);
    plot_callback(bac_100, res_100.minimizer, l, input_sample = 50:52, fig_name = "../graphics/res_100_int"*string(i, pad = 2)#=;ylims = (-0.5,0.5)=#)
end

# This file was generated using Literate.jl, https://github.com/fredrikekre/Literate.jl

