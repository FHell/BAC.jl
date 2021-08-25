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

##
mutable struct kuramoto_osc{w, N, theta, K}
    w::w
    N::N
    theta::theta
    K_av::K
end


function (dd::kuramoto_osc)(dx, x, i, p, t)
    # x -> Theta, p -> K
    #print(size(p))
    p = reshape(p,(dd.N,dd.N))
    p_total = sum(abs.(p))
    # dx .= [dd.w[k] + sum(p[k,j]/p_total*sin(x[k]-x[j]) for j in 1:dd.N)/dd.N for k in 1:dd.N]
    for k in 1:dd.N
        dx[k] = 0.
        for j in 1:k
            dx[k] -= abs(p[k,j]) * sin(x[k] - x[j])
            dx[j] += abs(p[k,j]) * sin(x[k] - x[j])
        end
        #=for j in 1:dd.N
            dx[k] -= abs(p[k,j]) * sin(x[k] - x[j])
        end=#
        dx[k] *= dd.K_av / p_total
        dx[k] += dd.w[k]
    end
    dx[1] += i
    # the relu(p) should be handles outside the function definition
    nothing
end

##
mutable struct KuramotoOutputMetric
    n_sys
    n_spec
end

function create_kuramoto_example(w, dim_sys, dim_spec, K,  tsteps, N_samples; modes=5)
    # scale down the input by a factor of 0.1
    f_spec = kuramoto_osc(w[dim_sys + 1:end], dim_spec, zeros(dim_spec), K)
    f_sys = kuramoto_osc(w[1:dim_sys], dim_sys, zeros(dim_sys), K)
    BAC_Loss(
        f_spec,
        f_sys,
        tsteps,
        (tsteps[1], tsteps[end]),
        [rand_fourier_input_generator(n, N=modes) for n = 1:N_samples], # input function i(t) 
        StandardOutputMetric(1, 1), # phase at interface node
        N_samples,
        dim_spec^2,
        dim_sys^2,
        zeros(dim_spec),
        zeros(dim_sys), 
        Tsit5()
    )
end

function (kur_met::KuramotoOutputMetric)(sol_sys, sol_spec)
    if sol_sys.retcode == :Success && sol_spec.retcode == :Success
        return sum(x -> x^2, (sol_sys[ last(kur_met.n_sys), :] .- sol_sys[first(kur_met.n_sys), :]) .- (sol_spec[ last(kur_met.n_spec), :] .- sol_spec[first(kur_met.n_spec), :]))
    else
        return Inf # Solvers failing is bad.
    end
end

##
##

dim_sys = 10
dim_spec = 2
N_samples = 10
K_sys_init = 6. * rand(dim_sys, dim_sys) .+ 1.
K_spec_init = 6. * rand(dim_spec, dim_spec) .+ 1.

p_initial = vcat(view(K_sys_init, 1:dim_sys^2), repeat(view(K_spec_init, 1:dim_spec^2), N_samples))

i = rand_fourier_input_generator(1)
##
# Try to solve differential equations without involving DiffEqFlux and BAC
omega = 3 * rand(dim_sys)
omega .-= mean(omega)
kur_ex = kuramoto_osc(omega, dim_sys, zeros(dim_sys), 50.)
kur_ex_spec = kuramoto_osc(omega, dim_spec, zeros(dim_spec), 25.)
res_spec = solve(ODEProblem((dy, y, p, t) -> kur_ex_spec(dy, y, i(t), p, t), ones(dim_spec), (0., 100.),  K_spec_init), Tsit5())

res_sys = solve(ODEProblem((dy, y, p, t) -> kur_ex(dy, y, i(t), p, t), ones(dim_sys), (0., 100.),  K_sys_init), Tsit5())

plot(res_sys)
plot(res_spec)
# savefig("../graphics/kuramoto_spec.png")

omega = ones(dim_sys + dim_spec)
omega .-= mean(omega)

@views begin
    p_syss = reshape(p_initial[1:dim_sys^2], (dim_sys, dim_sys))# p[1:bl.dim_sys, 1:bl.dim_sys]
    p_specs = [reshape(p_initial[(dim_sys^2 + 1 + (n - 1) * dim_spec^2):(dim_sys^2 + n * dim_spec^2)], (dim_spec, dim_spec)) for n in 1:N_samples]# [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec, bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
end

kur = create_kuramoto_example(omega, dim_sys, dim_spec, 20., 0.:100., N_samples)
solve_sys_spec(kur, i, p_syss, p_specs[1])


p_total = sum(abs.(p_syss))


## Check if kur(p) works - yes
scenarios = 1:3
sol1, sol2 = solve_bl_n(kur, 3, p_initial, scenario_nums=scenarios)
kur.output_metric(sol1, sol2)

l = kur(p_initial, abstol=1e-2, reltol=1e-2)

## For some reason using kur(p) inside an optimization loop results in an error. I have not been able to find the reason yet
# The error occurs in the differential equation system (line 36), as if p is a 4-element vector instead of 2x2 matrix.
# All the functions run normally without DiffEqFlux
res_10 = DiffEqFlux.sciml_train(
    p -> kur(p, abstol=1e-2, reltol=1e-2),
    p_initial,
    DiffEqFlux.ADAM(0.5),
    #DiffEqFlux.BFGS(),
    maxiters=5,
    cb=basic_bac_callback
    # cb = (p, l) -> plot_callback(kur, p, l, scenario_nums=scenarios)
    )

plot_callback(kur, p_initial, l, scenario_nums = 1)

##
using ForwardDiff
##
t = p -> kur(p, dim=2, abstol=1e-1, reltol=1e-1)

gradient(t, ones(160))
