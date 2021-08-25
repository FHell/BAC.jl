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

## The full system

mutable struct kuramoto_osc{w, N, K}
    w::w
    N::N
    K_av::K
end


function (dd::kuramoto_osc)(dx, x, i, p_in, t)
    # x -> Theta, p -> K
    p = reshape(p_in,(dd.N,dd.N))
    p_total = sum(relu.(p))
    for k in 1:dd.N
        dx[k] = 0.
        for j in 1:(k-1)
            dx[k] -= relu(p[k,j]) * sin(x[k] - x[j])
            dx[j] -= relu(p[j,k]) * sin(x[j] - x[k])
        end
    end
    for k in 1:dd.N
      dx[k] *= dd.K_av / p_total
      dx[k] += dd.w[k]
    end
    dx[1] += i
    nothing
end

## The specification, a kuramoto with inertia

function spec(dx, x, i, p, t)
    dx[1] = x[2] + p[1] * i
    dx[2] = relu(p[2]) - relu(p[3]) * x[2] + relu(p[4]) * i
    nothing
end

##

function create_kuramoto_example(w, dim_sys, K,  tsteps, N_samples; modes=5)
    f_sys = kuramoto_osc(w[1:dim_sys], dim_sys, K)
    BAC_Loss(
        spec,
        f_sys,
        tsteps,
        (tsteps[1], tsteps[end]),
        [rand_fourier_input_generator(n, N=modes) for n = 1:N_samples], # input function i(t) 
        StandardOutputMetric(1, 1), # phase at interface node
        N_samples,
        4, # Parameters in the spec
        dim_sys^2,
        zeros(2), # Initial conditions for spec
        zeros(dim_sys), 
        Tsit5()
    )
end

## Parameters

dim_sys = 10
N_samples = 10
p_sys_init = 6. * rand(dim_sys, dim_sys) .+ 1.
p_spec_init = rand(4)

p_initial = vcat(view(p_sys_init, 1:dim_sys^2), repeat(view(p_spec_init, 1:4), N_samples))

i = rand_fourier_input_generator(1)

##

plot(i, 0., 2pi)

##

omega = 3 * rand(dim_sys)
omega .-= mean(omega)
kur_ex = kuramoto_osc(omega, dim_sys, 50.)

res_spec = solve(ODEProblem((dy, y, p, t) -> spec(dy, y, 0., p, t), ones(2), (0., 2pi),  p_spec_init), Tsit5())
res_sys = solve(ODEProblem((dy, y, p, t) -> kur_ex(dy, y, 0., p, t), ones(dim_sys), (0., 2pi),  p_sys_init), Tsit5())

##

plot(res_sys)
##

plot(res_spec)

##

omega = rand(dim_sys)
omega .-= mean(omega)

@views begin
    p_syss = reshape(p_initial[1:dim_sys^2], (dim_sys, dim_sys))
    p_specs = [p_initial[(dim_sys^2 + 1 + (n - 1) * 4):(dim_sys^2 + n * 4)] for n in 1:N_samples]
end

kur = create_kuramoto_example(omega, dim_sys, 10., 0.:0.1:2pi, N_samples)

solve_sys_spec(kur, i, p_syss, p_specs[1])

## Check if kur(p) works - yes
scenarios = 1:3
sol1, sol2 = solve_bl_n(kur, 3, p_initial, scenario_nums=scenarios)
kur.output_metric(sol1, sol2)

l = kur(p_initial, abstol=1e-2, reltol=1e-2)

## Plot where we start

plot_callback(kur, p_initial, l, scenario_nums = 5)

## For some reason using kur(p) inside an optimization loop results in an error. I have not been able to find the reason yet
# The error occurs in the differential equation system (line 36), as if p is a 4-element vector instead of 2x2 matrix.
# All the functions run normally without DiffEqFlux
res_10 = DiffEqFlux.sciml_train(
    p -> kur(p, abstol=1e-4, reltol=1e-4),
    p_initial,
    # DiffEqFlux.ADAM(0.1),
    DiffEqFlux.BFGS(),
    maxiters=5,
    cb=basic_bac_callback
    # cb = (p, l) -> plot_callback(kur, p, l, scenario_nums=scenarios)
    )


##

plot_callback(kur, res_10.u, res_10.minimum, scenario_nums = 5)

##

res_iter = DiffEqFlux.sciml_train(
    p -> kur(p, abstol=1e-4, reltol=1e-4),
    res_10.u,
    # DiffEqFlux.ADAM(0.1),
    DiffEqFlux.BFGS(),
    maxiters=30,
    cb=basic_bac_callback
    # cb = (p, l) -> plot_callback(kur, p, l, scenario_nums=scenarios)
    )

##

plot_callback(kur, res_iter.u, res_iter.minimum, scenario_nums = 2)

##

res_iter = DiffEqFlux.sciml_train(
    p -> kur(p, abstol=1e-4, reltol=1e-4),
    res_iter.u,
    # DiffEqFlux.ADAM(0.1),
    DiffEqFlux.AMSGrad(0.01),
    maxiters=200,
    cb=basic_bac_callback
    # cb = (p, l) -> plot_callback(kur, p, l, scenario_nums=scenarios)
    )

##

plot_callback(kur, res_iter.u, res_iter.minimum, scenario_nums = 1)

##