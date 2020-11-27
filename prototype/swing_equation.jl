##
cd(@__DIR__)
using Pkg
Pkg.activate(".")
# Pkg.instantiate()

using Random
using DiffEqFlux
using OrdinaryDiffEq
using Plots
using LightGraphs
using Statistics
using ParameterizedFunctions, DiffEqDevTools
using DiffEqSensitivity

include("../src/Core.jl")
include("../src/ExampleSystems.jl")
include("../src/PlotUtils.jl")
include("../src/Benchmark.jl")

####################################################################################
## setup functions

struct swing_eq{T, TA, TN, TP, TK, TI}
    B::T
    B_trans::TA
    N::TN
    P::TP
    K::TK
    input_coupling::TI
end
# first N entries are phase, second half are frequency variables
function (dd::swing_eq)(dx, x, i, p, t)
    flows = dd.K .* dd.B * sin.(dd.B_trans * x[1:dd.N])
    @. dx[1:dd.N] = x[dd.N+1:2dd.N]
    # the relu(p) should be handles outside the function definition
    @. dx[dd.N+1:2dd.N] = p[dd.N+1:2dd.N] * ( dd.P - (0.1 + relu(p[1:dd.N])) * x[dd.N+1:2dd.N] - flows )
    dx[dd.N+1] -= dd.K * sin(x[1] - dd.input_coupling * i)
    nothing
end

struct swing_spec{TP, TK, TI, TD}
    P::TP
    K::TK
    D::TD
    input_coupling::TI
end
function (dd::swing_spec)(dx, x, i, p, t)
    @. dx[1:2] = x[3:4]
    # the relu(p) should be handles outside the function definition
    @. dx[3:4] = ( dd.P - dd.D * x[3:4] -  dd.K .* sin.([x[1] - x[2], x[2] - x[1]]) ) * (0.5 + relu(p))
    dx[3] -= dd.K * sin(x[1] - dd.input_coupling * i)
    nothing
end

struct single_spec{TP, TK, TI, TD}
    P::TP
    K::TK
    D::TD
    input_coupling::TI
end
function (dd::single_spec)(dx, x, i, p, t)
    dx[1] = x[2]
    # the relu(p) should be handles outside the function definition
    dx[2] = (relu(p[1]) + 1.) * (dd.P - (relu(p[2]) + 0.1) * x[2] - dd.K * sin(x[1] - dd.input_coupling * i))
    nothing
end


mutable struct SwingOutputMetric
    n_sys
    n_spec
end
function (som::SwingOutputMetric)(sol_sys, sol_spec)
    if sol_sys.retcode == :Success && sol_spec.retcode == :Success
        # match the first two phase angles to match the flow on the first link
        # we take the phase at the first node as reference, so ϕ₁≡0
        # the phase at the second node is then ϕ₂ = sol[3, :] .- sol[1, :]
        # the loss function is then ∑ (ϕ₂-ϕ₂')^2 where ϕ₂' is the phase diff in the spec
        return 1. - 1. / (sum(x->x^2, sol_sys[1, :] .- sol_spec[1, :]) + 1.)
    else
        return Inf # Solvers failing is bad.
    end
end

function create_swing_example(dim_sys, dim_spec, av_deg, tsteps, N_samples; modes=4)
    g_sys = barabasi_albert(dim_sys, av_deg)

    @assert dim_sys == nv(g_sys)

    # scale down the input by a factor of 0.1
    f_spec = single_spec(0., 8., 0.1, 0.1)
    B_sys = incidence_matrix(g_sys, oriented=true)
    f_sys = swing_eq(B_sys, B_sys', dim_sys, (-1).^(1:dim_sys), 8., 0.1)

    BAC_Loss(
        f_spec,
        f_sys,
        tsteps,
        (tsteps[1], tsteps[end]),
        [rand_fourier_input_generator(n, modes) for n = 1:N_samples], # input function i(t) 
        SwingOutputMetric([1, 3], [1, 3]), # flow on first line
        #StandardOutputMetric(1, 1), # phase at interface node
        N_samples,
        dim_spec,
        2*dim_sys,
        zeros(2dim_spec),
        zeros(2dim_sys),
    )
end

#####################################################################
## setup BAC problem

# create the system
n_nodes = 10
dim_sys = 2 * n_nodes
dim_spec = 2
N_samples = 10

bac_10 = create_swing_example(n_nodes, dim_spec, 3, 0.:0.1:10., N_samples)

p_initial = ones(dim_sys + dim_spec * N_samples);

##
 
# we can easily plot the input sample
plot(bac_10.tsteps, bac_10.input_sample, c=:gray, alpha=1, legend=false)

##

# let's look at a system trajectory
sol1, sol2 = solve_bl_n(bac_10, 3, p_initial);
@show bac_10.output_metric(sol1, sol2)

# plot frequency
plot(sol1, vars=11:dim_sys, c=:black, legend=false)
plot!(sol2, vars=1:dim_spec, c=:red, legend=false)


##

l = bac_10(p_initial) 
l = bac_10(p_initial; abstol=1e-5, reltol=1e-5) 
il = individual_losses(bac_10, p_initial)

# TODO adjust callback to plot flow ?
plot_callback(bac_10, p_initial, l)

####################################################################################
## test optimisation

# Train with 10 samples, low accuracy and relatively large ADAM step size: (1.5 minutes on my Laptop)
@time res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p; abstol=1e-5, reltol=1e-5, sensealg=DiffEqSensitivity.ForwardDiffSensitivity()),
    p_initial,
    DiffEqFlux.AMSGrad(0.1),
    maxiters = 5,
    cb = basic_bac_callback
    #cb = (p, l) -> plot_callback(bac_10, p, l, 1)
)

@show bac_10(res_10.minimizer, abstol=1e-5, reltol=1e-5) # 7279.409305626946

##
# Train with 10 samples, low accuracy and relatively large ADAM step size: (1.5 minutes on my Laptop)

bac_10 = resample(rand_fourier_input_generator, bac_10)
@time res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p; abstol=1e-6, reltol=1e-6, maxiters=1E6, sensealg=DiffEqSensitivity.ForwardDiffSensitivity()),
    res_10.minimizer,
    DiffEqFlux.AMSGrad(0.1),
    maxiters = 50,
    cb = (p, l) -> plot_callback(bac_10, p, l, 1)
)
##

@show bac_10(res_10.minimizer, abstol=1e-6, reltol=1e-6, maxiters=1E6) #5096.921157700889

for i in 1:10
plot_callback(bac_10, res_10.minimizer, 0., i)
end

##

############### We were testing this until here ##################

##
# Train with 10 samples, low accuracy and relatively large ADAM step size: (1.5 minutes on my Laptop)
@time res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p; abstol=1e-8, reltol=1e-8),
    res_10.minimizer,
    DiffEqFlux.ADAM(0.5),
    maxiters = 10,
    cb = (p, l) -> plot_callback(bac_10, p, l, 1)
)

@show bac_10(res_10.minimizer) # 5073.413235042178

##

# let's look at a system trajectory again
sol1, sol2 = solve_bl_n(bac_10, rand(1:10), res_10.minimizer)

@show bac_10.output_metric(sol1, sol2)

# plot frequency
plot(sol1[1, :] .- sol1[3, :], c=:black, legend=false)
plot!(sol2[1, :] .- sol2[3, :], c=:red, legend=false)

##
# We can check the quality of the resulting minimizer by optimizing the specs only (a much simpler repeated 2d optimization problem)
p2 = bac_spec_only(bac_10, res_10.minimizer)

@show bac_10(p2) # 5070.929431347097

@show individual_losses(bac_10, p2) 

##

# In order to understand how much we were overfitting with respect to the concrete sample, we resample
# That is, we generate a new problem with a new sample from the same distribution:
bac_10_rs = resample(rand_fourier_input_generator, bac_10)
p3 = bac_spec_only(bac_10_rs, p2)

@show bac_10(p3)

##
# this got it down to 0.01 loss. we can look at the minimizer:

# p_sys
@show res_10.minimizer[1:dim_sys] |> relu |> println
# [3.1004914201819593, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# p_spec
@show res_10.minimizer[dim_sys+1:dim_sys+dim_spec*N_samples] |> relu |> println
# [21.724647422980105, 1.1327810492139974, 21.488022770061658, 1.2955449338990668, 29.693535400360048, 0.0, 5.28509017931121, 0.0, 3.828808060232345, 0.0, 0.0, 0.0, 31.189165828210648, 0.0, 24.646260154003162, 0.02722732297887498, 0.0, 0.0, 22.17688419187349, 1.0810186317295836, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

####################################################################################
## benchmark


# Implement training with a set of optimizers
setups = [Dict(:opt=>DiffEqFlux.ADAM(0.1), :name=>"ADAM(0.1)"), # Add support for different maxiters?
          Dict(:opt=>DiffEqFlux.Descent(0.1), :name=>"Descent(0.1)"),
          Dict(:opt=>DiffEqFlux.AMSGrad(0.1), :name=>"AMSGrad(0.1)"),
          Dict(:opt=>DiffEqFlux.NelderMead(), :name=>"NelderMead()"),
          # Dict(:opt=>DiffEqFlux.BFGS(initial_stepnorm = 0.01), :name=>"BFGS(initial_stepnorm = 0.01)"),
          # Dict(:opt=>DiffEqFlux.BFGS(initial_stepnorm = 0.1), :name=>"BFGS(initial_stepnorm = 0.1)"), #seems to get stuck
          Dict(:opt=>DiffEqFlux.MomentumGradientDescent(), :name=>"MomentumGradientDescent()")]

t, l = train_set(1, setups, bac_10, p_initial) #Compiling everything
t, l = train_set(20, setups, bac_10, p_initial)



