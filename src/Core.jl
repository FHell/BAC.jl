export BAC_Problem
import Base.@kwdef

@kwdef mutable struct BAC_Problem
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
    solver = Tsit5()
end

function (bl::BAC_Problem)(p; solver_options...)
    # Evalute the loss function of the BAC problem

    @views begin
        p_sys = p[1:bl.dim_sys]
        p_specs = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
    end

    loss = 0.

    for n in 1:bl.N_samples
        i = bl.input_sample[n]
        loss += bl.output_metric(solve_sys_spec(bl, i, p_sys, p_specs[n]; solver_options...)...)
    end

    loss
end

function (bl::BAC_Problem)(n, p_sys, p_spec; solver_options...)
    # loss function for sample n only
    i = bl.input_sample[n]
    bl.output_metric(solve_sys_spec(bl, i, p_sys, p_spec; solver_options...)...)
end

function solve_sys_spec(bl, i, p_sys, p_spec; solver_options...)
    dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), bl.solver; saveat=bl.tsteps, solver_options...)
    dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_spec), bl.solver; saveat=bl.tsteps, solver_options...)
    dd_sys, dd_spec
end

function solve_bl_n(bl, n::Int, p; solver_options...)
    @views begin
        p_sys = p[1:bl.dim_sys]
        p_spec = p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec]
    end

    i = bl.input_sample[n]

    solve_sys_spec(bl, i, p_sys, p_spec; solver_options...)
end


function bac_spec_only(bl::BAC_Problem, p_initial; optimizer=DiffEqFlux.ADAM(0.01), optimizer_options=(:maxiters => 100,), solver_options...)
    # Optimize the specs only

    p = copy(p_initial)
    @views begin
        p_sys = p[1:bl.dim_sys]
        p_specs = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
    end

    for n in 1:bl.N_samples
    println(n)
    res = DiffEqFlux.sciml_train(
        x -> bl(n, p_sys, x), # Loss function for each n individually
        Array(p_specs[n]),
        optimizer;
        optimizer_options...)
        p_specs[n] .= res.minimizer
    end

    p
end


function individual_losses(bl::BAC_Problem, p)
    # Return the array of losses
    @views begin
        p_sys = p[1:bl.dim_sys]
        p_specs = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
    end

    [bl(n, p_sys, p_specs[n]) for n in 1:bl.N_samples]
end


# A constructors for BAC_Problem with standard solver

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


# Resampling the BAC_Problem
export resample
function resample(sampler, bac::BAC_Problem)
    new_input_sample = [sampler(n) for n in 1:bac.N_samples]
    BAC_Problem(bac.f_spec, bac.f_sys, bac.tsteps, bac.t_span, new_input_sample, bac.output_metric, bac.N_samples, bac.dim_spec, bac.dim_sys, bac.y0_spec, bac.y0_sys, bac.solver)
end

# Basic callback

export basic_bac_callback
function basic_bac_callback(p, loss)
    display(loss)
    # Tell sciml_train to not halt the optimization. If return true, then
    # optimization stops.
    return false
end
