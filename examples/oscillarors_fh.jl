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
using LinearAlgebra

include("../src/Core.jl")
include("../src/ExampleSystems.jl")
include("../src/PlotUtils.jl")
include("../src/Benchmark.jl")

Random.seed!(1) # For reproducability fix the seed

## The full system

mutable struct kuramoto_osc{w, N, K}
    w::w
    N::N
    K_av::K
end


function (dd::kuramoto_osc)(dx, x, i, p, t)
    # x -> Theta, p -> K
    K_total = 0.
    p_ind = 1
    for k in 1:dd.N
        dx[k] = x[k + dd.N]
    end
    for k in 1:dd.N
        for j in 1:(k-1)
            dx[k+dd.N] -= (relu(p[p_ind]) + 1.) * sin(x[k] - x[j])
            dx[j+dd.N] -= (relu(p[p_ind]) + 1.) * sin(x[j] - x[k])
            K_total += relu(p[p_ind]) + 1.
            p_ind += 1
        end
    end
    for k in 1:dd.N
      dx[k+dd.N] *= dd.K_av / K_total * (p_ind - 1) # (p_ind - 1) counts the number of edges
      dx[k+dd.N] += dd.w[k] - (relu(p[p_ind]) + 1.) * x[k+dd.N]
      p_ind += 1
    end
    # dx[1+dd.N] += dd.K_av * i
    dx[1+dd.N] += 1. * i
    nothing
end
## The specification, a kuramoto with inertia

function spec(dx, x, i, p, t)
    dx[1] = x[2]
    dx[2] = p[2] - (relu(p[3]) + 1.) * x[2]  + 1. * i# inertial node with a slackbus
    nothing
end


function create_kuramoto_example(w, N_osc, K,  tsteps, N_samples; modes=5)
    f_sys = kuramoto_osc(w[1:N_osc], N_osc, K)
    BAC_Loss(
        spec,
        f_sys,
        tsteps,
        (tsteps[1], tsteps[end]),
        [rand_fourier_input_generator(n, N=modes) for n = 1:N_samples], # input function i(t) 
        out_metric, # phase at interface node
        N_samples,
        3, # Parameters in the spec
        N_osc * (N_osc - 1) ÷ 2 + N_osc, # Parameters in the sys
        zeros(2), # Initial state conditions for spec
        zeros(2 * N_osc), # Initial state conditions for sys
        Tsit5()
    )
end

##

const t_steps = 0.:0.1:4pi
const n_transient = length(0.:0.1:2pi)

##

function out_metric(sol_sys, sol_spec)
    if sol_sys.retcode == :Success && sol_spec.retcode == :Success
        return sum((sol_sys[1, n_transient:end] .- sol_spec[1, n_transient:end]) .^ 2)
    else
        return Inf # Solvers failing is bad.
    end
end


## Parameters

N_osc = 10
dim_p = N_osc * (N_osc - 1) ÷ 2 + N_osc
N_samples = 10
p_sys_init = 6. * rand(dim_p) .+ 1.
p_spec_init = rand(3)

p_initial = vcat(view(p_sys_init, 1:dim_p), repeat(view(p_spec_init, 1:3), N_samples))

i = rand_fourier_input_generator(1)

##

plot(i, 0., 4pi)

##

omega = 8. * randn(N_osc)
omega .-= mean(omega)

##
K_av = 1.
kur_ex = kuramoto_osc(omega, N_osc, K_av)

res_spec = solve(ODEProblem((dy, y, p, t) -> spec(dy, y, 0., p, t), ones(2), (t_steps[1], t_steps[end]),  p_spec_init), Tsit5())
res_sys = solve(ODEProblem((dy, y, p, t) -> kur_ex(dy, y, 0., p, t), ones(2*N_osc), (t_steps[1], t_steps[end]),  p_sys_init), Tsit5())
##

plot(res_sys)
##

plot(res_spec)

##

omega = 20. * randn(N_osc)
omega .-= mean(omega)

@views begin
    p_syss = p_initial[1:dim_p]
    p_specs = [p_initial[(dim_p + 1 + (n - 1) * 3):(dim_p + n * 3)] for n in 1:N_samples]
end

kur = create_kuramoto_example(omega, N_osc, K_av, t_steps, N_samples)

solve_sys_spec(kur, i, p_syss, p_specs[1])

## Check if kur(p) works - yes
scenarios = 1:3
sol1, sol2 = solve_bl_n(kur, 3, p_initial, scenario_nums=scenarios)
kur.output_metric(sol1, sol2)



## Plot where we start

p_initial = bac_spec_only(kur, p_initial; optimizer_options=(:maxiters => 1000,), solver_options = (abstol = 1e-4, reltol=1e-4))

l = kur(p_initial, abstol=1e-4, reltol=1e-4)
plot_callback(kur, p_initial, l, scenario_nums = 1:10, plot_options = (:xlims => (kur.t_span[2]/2, kur.t_span[2])))
plot_callback(kur, p_initial, l, scenario_nums = 1:10, xlims = (kur.t_span[2]/2, kur.t_span[2]))

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

plot_callback(kur, res_1.u, res_1.minimum, scenario_nums = 10, xlims = (kur.t_span[2]/2, kur.t_span[2]))

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


plot_callback(kur, res_2.u, res_2.minimum, scenario_nums = 1:10, xlims = (kur.t_span[2]/2, kur.t_span[2]))

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

plot_callback(kur, res_3.u, res_3.minimum, scenario_nums = 5, xlims = (kur.t_span[2]/2, kur.t_span[2]))

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

plot_callback(kur, res_4.u, res_4.minimum, scenario_nums = 5, xlims = (kur.t_span[2]/2, kur.t_span[2]))

##

@views begin
    p_final = res_4.u[1:dim_p]
    p_final_specs = [res_4.u[(dim_p + 1 + (n - 1) * 3):(dim_p + n * 3)] for n in 1:N_samples]
end

##
scen = 1:5

plot_callback(kur, res_1.u, res_1.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))
plot_callback(kur, res_2.u, res_2.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))
plot_callback(kur, res_3.u, res_3.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))
plot_callback(kur, res_4.u, res_4.minimum, scenario_nums = scen, xlims = (kur.t_span[2]/2, kur.t_span[2]))

##


res_5 = DiffEqFlux.sciml_train(
    p -> kur(p, abstol=1e-4, reltol=1e-4),
    res_5.u,
    # DiffEqFlux.ADAM(0.1),
    DiffEqFlux.AMSGrad(0.01),
    # DiffEqFlux.BFGS(),
    maxiters=225,
    cb=basic_bac_callback
    # cb = (p, l) -> plot_callback(kur, p, l, scenario_nums=scenarios)
    )

##

plot_callback(kur, res_5.u, res_5.minimum, scenario_nums = 1:10, xlims = (kur.t_span[2]/2, kur.t_span[2]))


##
