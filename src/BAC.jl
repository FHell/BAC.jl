module BAC

using OrdinaryDiffEq
using DiffEqFlux
include("ExampleSystems.jl")
# import .ExampleSystems

include("PlotUtils.jl")
# import .PlotUtils

export BAC_Problem

struct BAC_Problem
  f_spec # f(dy, y, i, p, t)
  f_sys
  tsteps
  t_span
  input_sample # this is called with the sample as an argument and needs to return an input function i(t)
  output_metric
  N_samples::Int
  dim_spec::Int
  dim_sys::Int
  y0_spec
  y0_sys
  solver
end

function (bl::BAC_Problem)(p)
    # Evalute the loss function of the BAC problem

    @views begin
        p_sys = p[1:bl.dim_sys]
        p_specs = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
    end

    loss = 0.

    # For plotting evaluate the first sample outside the loop:
    n = 1
    i = bl.input_sample[n]
    dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), bl.solver; saveat=bl.tsteps)
    dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_specs[n]), bl.solver, saveat=bl.tsteps)
    loss += bl.output_metric(dd_sys, dd_spec)

    for n in 2:bl.N_samples
        i = bl.input_sample[n]
        dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), bl.solver, saveat=bl.tsteps)
        dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_specs[n]), bl.solver, saveat=bl.tsteps)
        loss += bl.output_metric(dd_sys, dd_spec)
    end

    loss, dd_sys, dd_spec
end


function (bl::BAC_Problem)(n, p_sys, p_spec)
    i = bl.input_sample[n]
    dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), bl.solver; saveat=bl.tsteps)
    dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_spec), bl.solver, saveat=bl.tsteps)
    
    loss = bl.output_metric(dd_sys, dd_spec)

    loss, dd_sys, dd_spec
end

function bac_spec_only(bl::BAC_Problem, p)
    # Optimize the specs only
    @views begin
        p_sys = p[1:bl.dim_sys]
        p_specs = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
    end

    p_min = similar(p)
    @views begin
        p_mins = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
    end

    for n in 1:bl.N_samples
        res = DiffEqFlux.sciml_train(
        p -> bl(n, p_sys, p),
        p_specs[n],
        DiffEqFlux.BFGS(initial_stepnorm = 0.01),
        maxiters = 100)
        p_mins[n] = res.minimizer
    end

    p_min
end

function individual_loss(bl::BAC_Problem, p_sys, p_specs, n)
    # Evalute the individual loss contributed by the nth sample
    i = bl.input_sample[n]
    dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), bl.solver, saveat=bl.tsteps)
    dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_specs[n]), bl.solver, saveat=bl.tsteps)
    bl.output_metric(dd_sys, dd_spec)
end

export individual_losses
function individual_losses(bl::BAC_Problem, p)
    # Return the array of losses
    @views begin
        p_sys = p[1:bl.dim_sys]
        p_specs = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
    end

    [individual_loss(bl::BAC_Problem, p_sys, p_specs, n) for n in 1:bl.N_samples]
end


# The constructors for BAC_Problem

function BAC_Problem(
    f_spec,
    f_sys,
    tsteps,
    t_span,
    input_sample, # this is called with the sample as an argument and needs to return an input function i(t)
    output_metric,
    N_samples::Int,
    dim_spec::Int,
    dim_sys::Int,
    y0_spec,
    y0_sys; solver = Tsit5())

    BAC_Problem(f_spec, f_sys, tsteps, t_span, input_sample, output_metric, N_samples, dim_spec, dim_sys, y0_spec, y0_sys, solver)
end

function BAC_Problem(
    f_spec,
    f_sys,
    tsteps,
    input_sample, # this is called with the sample as an argument and needs to return an input function i(t)
    output_metric,
    N_samples::Int,
    dim_spec::Int,
    dim_sys::Int; solver = Tsit5())
    t_span = (tsteps[1], tsteps[end])
    y0_spec = zeros(dim_spec)
    y0_sys = zeros(dim_sys)
    BAC_Problem(f_spec, f_sys, tsteps, t_span, input_sample, output_metric, N_samples, dim_spec, dim_sys, y0_spec, y0_sys, solver)
 end

# Resampling the BAC_Problem
export resample
function resample(sampler, bac::BAC_Problem)
    new_input_sample = [sampler(n) for n in 1:bac.N_samples]
    BAC_Problem(bac.f_spec, bac.f_sys, bac.tsteps, bac.t_span, new_input_sample, bac.output_metric, bac.N_samples, bac.dim_spec, bac.dim_sys, bac.y0_spec, bac.y0_sys, bac.solver)
end

# Basic callbacks

export basic_bac_callback

function basic_bac_callback(p, loss, dd_sys, dd_spec)
    display(loss)
    # Tell sciml_train to not halt the optimization. If return true, then
    # optimization stops.
    return false
end


end # module
