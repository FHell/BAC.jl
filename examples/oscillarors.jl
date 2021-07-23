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

include("src/Core.jl")
include("src/ExampleSystems.jl")
include("src/PlotUtils.jl")
include("src/Benchmark.jl")

mutable struct kuramoto_osc{K, N, theta} #mutable for the development
    K::K
    N::N
    theta::theta
end

function (dd::kuramoto_osc)(dx, x, i, p, t)
    # x -> Theta, p -> omega
    dx = [p[k] + K*sum(sin(dd.x[k]-dd.x[j]) for j in 1:dd.N)/N for k in 1:dd.N]
    # the relu(p) should be handles outside the function definition
    nothing
end

mutable struct KuramotoOutputMetric
    n_sys
    n_spec
end

function create_kuramoto_example(dim_sys, dim_spec, tsteps, N_samples; modes = 5)
    #K_sys = rand(dim_sys) - if K was an optimization parameter
    K = 1
    # scale down the input by a factor of 0.1
    f_spec = kuramoto_osc(K, dim_spec, zeros(dim_spec))
    f_sys = kuramoto_osc(K, dim_sys, zeros(dim_sys))
    BAC_Loss(
        f_spec,
        f_sys,
        tsteps,
        (tsteps[1], tsteps[end]),
        [rand_fourier_input_generator(n, modes) for n = 1:N_samples], # input function i(t) 
        StandardOutputMetric(1, 1), # phase at interface node
        N_samples,
        dim_spec,
        dim_sys,
        zeros(2dim_spec),
        zeros(2dim_sys),
    )
end

function (kur_met::KuramotoOutputMetric)(sol_sys, sol_spec)
    if sol_sys.retcode == :Success && sol_spec.retcode == :Success
        # match the first two phase angles to match the flow on the first link
        # we take the phase at the first node as reference, so ϕ₁≡0
        # the phase at the second node is then ϕ₂ = sol[3, :] .- sol[1, :]
        # the loss function is then ∑ (ϕ₂-ϕ₂')^2 where ϕ₂' is the phase diff in the spec
        return sum(x->x^2, (sol_sys[ last(kur_met.n_sys), :] .- sol_sys[first(kur_met.n_sys), :]) .- (sol_spec[ last(kur_met.n_spec), :] .- sol_spec[first(kur_met.n_spec), :]) )
    else
        return Inf # Solvers failing is bad.
    end
end

kur = create_kuramoto_example(10,2,100,10)

p_initial = ones(dim_sys + dim_spec * N_samples);

plot(kur.tsteps, kur.input_sample, c=:gray, alpha=1, legend=false)

