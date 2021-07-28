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

include("src/Core.jl")
include("src/ExampleSystems.jl")
include("src/PlotUtils.jl")
include("src/Benchmark.jl")

mutable struct kuramoto_osc{w, N, theta, K}
    w::w
    N::N
    theta::theta
    K_av::K
end



function (dd::kuramoto_osc)(dx, x, i, p, t)
    # x -> Theta, p -> omega
    p_total = sum(abs.(p))
    #dx .= [dd.w[k] + sum(p[k,j]/p_total*sin(x[k]-x[j]) for j in 1:dd.N)/dd.N for k in 1:dd.N]
    for k in 1:dd.N
        dx[k] = 0.
        for j in 1:k
            dx[k]-=abs(p[k,j])*sin(x[k]-x[j])
            dx[j]+=abs(p[k,j])*sin(x[k]-x[j])
        end
        dx[k] *= dd.K_av/p_total
        dx[k] += dd.w[k]
    end
    dx[1] += i
    # the relu(p) should be handles outside the function definition
    nothing
end

##

dim_sys = 10
dim_spec = 2
N_samples = 10
K_sys_init = 3*rand(dim_sys,dim_sys).+1.

i = rand_fourier_input_generator(1)
##
omega = 3*rand(dim_sys)
omega .-= mean(omega)
kur_ex = kuramoto_osc(omega, dim_sys, zeros(dim_sys), 50.)

dd_sys = solve(ODEProblem((dy,y,p,t) -> kur_ex(dy, y, i(t), p, t), ones(dim_sys), (0.,100.),  K_sys_init), Tsit5())
#=dy0 = zeros(dim_sys)
y0 = ones(dim_sys)
kur_ex(dy0, y0, i(0.),K_sys_init, 0.)=#
plot(dd_sys)
#=
solve_sys_spec(kur_ex, rand_fourier_input_generator(1), p_initial[1:dim_sys, 1:dim_sys], p_initial[dim_sys:end, dim_sys:end])
=#
##


##
mutable struct KuramotoOutputMetric
    n_sys
    n_spec
end

function create_kuramoto_example(w, dim_sys, dim_spec, K,  tsteps, N_samples; modes = 5)
    #K_sys = rand(dim_sys) - if K was an optimization parameter
    # scale down the input by a factor of 0.1
    f_spec = kuramoto_osc(w[dim_sys+1:end], dim_spec, zeros(dim_spec), K)
    f_sys = kuramoto_osc(w[1:dim_sys], dim_sys, zeros(dim_sys), K)
    BAC_Loss(
        f_spec,
        f_sys,
        tsteps,
        (tsteps[1], tsteps[end]),
        [rand_fourier_input_generator(n, N = modes) for n = 1:N_samples], # input function i(t) 
        StandardOutputMetric(1, 1), # phase at interface node
        N_samples,
        dim_spec,
        dim_sys,
        zeros(dim_spec),
        zeros(dim_sys),
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

function (bl::BAC_Loss)(p; dim =1, solver_options...)
    # Evalute the loss function of the BAC problem
    if dim == 1
        @views begin
            p_sys = p[1:bl.dim_sys]
            p_specs = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
        end

        loss = 0.

        for n in 1:bl.N_samples
            i = bl.input_sample[n]
            loss += bl.output_metric(solve_sys_spec(bl, i, p_sys, p_specs[n]; solver_options...)...)
        end

        loss / bl.N_samples
    elseif dim == 2
        @views begin
            p_sys = p[1:bl.dim_sys, 1:bl.dim_sys]
            p_specs = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec, bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
        end

        loss = 0.

        for n in 1:bl.N_samples
            i = bl.input_sample[n]
            loss += bl.output_metric(solve_sys_spec(bl, i, p_sys, p_specs[n]; solver_options...)...)
        end

        loss / bl.N_samples
    end
end

dim_sys = 10
dim_spec = 2
N_samples = 10
omega = ones(dim_sys+dim_spec)
kur = create_kuramoto_example(omega,dim_sys,dim_spec,0.:100.,N_samples, 5.)
K_sys_init = 3*ones(dim_sys,dim_sys)
K_spec_init = 3*ones(dim_spec,dim_spec)
p_initial = [K_sys_init zeros(dim_sys, N_samples*dim_spec); zeros(N_samples*dim_spec, dim_sys+N_samples*dim_spec)]

for i in 1:N_samples
    p_initial[(dim_sys+dim_spec*i-1):(dim_sys+dim_spec*i), (dim_sys+dim_spec*i-1):(dim_sys+dim_spec*i)] = K_spec_init
end

kur(p_initial, dim = 2)
#plot(kur.tsteps, kur.input_sample, c=:gray, alpha=1, legend=false)
kur.f_sys(K_sys_init)

solve_sys_spec(kur, rand_fourier_input_generator(1), p_initial[1:dim_sys, 1:dim_sys], p_initial[dim_sys:end, dim_sys:end])

res_10 = DiffEqFlux.sciml_train(
    p -> kur(p; abstol=1e-2, reltol=1e-2),
    p_initial,
    DiffEqFlux.ADAM(0.5),
    maxiters = 5,
    cb = basic_bac_callback
    #cb = (p, l) -> plot_callback(kur, p, l, scenario_nums=scenarios)
    )

#=
p = zeros(6*6 + 2*2)
p_1 = view(p, 1:6*6)
p_2 = view(p, 6*6 + 1:6*6 + 2*2)
##

p_2d_1 = reshape(p_1, (6,6))
p_2d_2 = reshape(p_2, (2,2))

=# 