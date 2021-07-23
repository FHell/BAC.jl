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

function create_curamoto_example(dim_sys, dim_spec, tsteps, N_samples)
    #K_sys = rand(dim_sys)
    @assert dim_sys == nv(g_sys)
    # scale down the input by a factor of 0.1
    f_spec = kuramoto_osc(K, dim_spec, zeros(dim_spec))
    B_sys = incidence_matrix(g_sys, oriented=true)
    f_sys = swing_eq(K, dim_sys, zeros(dim_sys))

    BAC_Loss(
        f_spec,
        f_sys,
        tsteps,
        (tsteps[1], tsteps[end]),
        [BAC.rand_fourier_input_generator(n, modes) for n = 1:N_samples], # input function i(t) 
        BAC.StandardOutputMetric(1, 1), # phase at interface node
        N_samples,
        dim_spec,
        dim_sys,
        zeros(2dim_spec),
        zeros(2dim_sys),
    )
end