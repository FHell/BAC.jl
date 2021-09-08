cd(@__DIR__)
cd("..")

using Pkg
Pkg.activate(".")

using BAC

Random.seed!(1);
using Pipe: @pipe
using LaTeXStrings
using Statistics
using DiffEqFlux

##
const t_steps = 0.:0.1:4pi
const n_transient = length(0.:0.1:2pi)
##

dim_p_spec = 2
N_osc = 10
dim_p = N_osc * (N_osc - 1) ÷ 2 + N_osc
N_samples = 10
p_sys_init = 6. * rand(dim_p) .+ 1.
p_spec_init = rand(dim_p_spec) .+ 1.

p_initial = vcat(p_sys_init, repeat(p_spec_init, N_samples))

i = BAC.rand_fourier_input_generator(1)
K_av = 1.

##
plot(i, 0., 4pi)

##

omega = 1. * randn(N_osc);
omega .-= mean(omega)

##

@views begin
    p_syss = p_initial[1:dim_p]
    p_specs = [p_initial[(dim_p + 1 + (n - 1) * dim_p_spec):(dim_p + n * dim_p_spec)] for n in 1:N_samples]
end

kur = BAC.create_kuramoto_example(omega, N_osc, dim_p_spec, K_av, t_steps, N_samples) # specify modes = 0 for no input

solve_sys_spec(kur, i, p_syss, p_specs[1])

scen = 1:5
sol1, sol2 = BAC.solve_bl_n(kur, 3, p_initial, scenario_nums = scen) # n and scen_nums???
kur.output_metric(sol1, sol2)

## Plot where we start
p_initial = BAC.bac_spec_only(kur, p_initial; optimizer_options=(:maxiters => 1000,), solver_options = (abstol = 1e-4, reltol=1e-4))

l = kur(p_initial, abstol=1e-4, reltol=1e-4)
plot_callback(kur, p_initial, l, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))
##

## For some reason using kur(p) inside an optimization loop results in an error. I have not been able to find the reason yet
# The error occurs in the differential equation system (line 36), as if p is a 4-element vector instead of 2x2 matrix.
# All the functions run normally without DiffEqFlux
res_1 = DiffEqFlux.sciml_train(
    p -> kur(p, abstol=1e-4, reltol=1e-4),
    p_initial,
    DiffEqFlux.ADAM(0.1),
    # DiffEqFlux.BFGS(),
    maxiters=25,
    cb=basic_bac_callback
    # cb = (p, l) -> plot_callback(kur, p, l, scenario_nums=scenarios)
    )

##

plot_callback(kur, res_1.u, res_1.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))

##

res_2 = DiffEqFlux.sciml_train(
    p -> kur(p, abstol=1e-4, reltol=1e-4),
    res_1.u,
    # DiffEqFlux.ADAM(0.1),
    DiffEqFlux.BFGS(),
    maxiters=25,
    cb=basic_bac_callback
    # cb = (p, l) -> plot_callback(kur, p, l, scenario_nums=scenarios)
    )

##

plot_callback(kur, res_2.u, res_2.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))

##

res_3 = DiffEqFlux.sciml_train(
    p -> kur(p, abstol=1e-4, reltol=1e-4),
    res_2.u,
    # DiffEqFlux.BFGS(),
    # DiffEqFlux.ADAM(0.1),
    DiffEqFlux.AMSGrad(0.01),
    maxiters=100,
    cb=basic_bac_callback
    # cb = (p, l) -> plot_callback(kur, p, l, scenario_nums=scenarios)
    )

##

plot_callback(kur, res_3.u, res_3.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))

##

res_4 = DiffEqFlux.sciml_train(
    p -> kur(p, abstol=1e-4, reltol=1e-4),
    res_3.u,
    # DiffEqFlux.ADAM(0.1),
    DiffEqFlux.BFGS(),
    maxiters=25,
    cb=basic_bac_callback
    # cb = (p, l) -> plot_callback(kur, p, l, scenario_nums=scenarios)
    )

##

plot_callback(kur, res_4.u, res_4.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))

##

plot_callback(kur, res_1.u, res_1.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))
plot_callback(kur, res_2.u, res_2.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))
plot_callback(kur, res_3.u, res_3.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))
plot_callback(kur, res_4.u, res_4.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))

##

res_5 = DiffEqFlux.sciml_train(
    p -> kur(p, abstol=1e-4, reltol=1e-4),
    res_4.u,
    # DiffEqFlux.ADAM(0.1),
    DiffEqFlux.AMSGrad(0.01),
    # DiffEqFlux.BFGS(),
    maxiters=225,
    cb=basic_bac_callback
    # cb = (p, l) -> plot_callback(kur, p, l, scenario_nums=scenarios)
    )

##

plot_callback(kur, res_5.u, res_5.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))

##

@views begin
    p_final = res_5.u[1:dim_p]
    p_final_specs = [res_5.u[(dim_p + 1 + (n - 1) * dim_p_spec):(dim_p + n * dim_p_spec)] for n in 1:N_samples]
end

##

kur_100 = resample(BAC.rand_fourier_input_generator, kur; n = 100);

p_sys_init_100 = 6. * rand(dim_p) .+ 1.
p_spec_init_100 = rand(dim_p_spec) .+ 1.

p_100 = vcat(p_sys_init, repeat(p_spec_init, 100))

p_100[1:dim_p] .= relu.(p_final)
p_100[dim_p+1:end] .= repeat(relu.(p_final_specs[1]), 100)

p_initial_100 = BAC.bac_spec_only(kur_100, p_100)