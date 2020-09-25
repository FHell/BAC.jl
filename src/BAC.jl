module BAC

using OrdinaryDiffEq
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
end

function (bl::BAC_Problem)(p)
    # Evalute the loss function of the BAC problem

    @views begin
        p_sys = p[1:bl.dim_sys]
        p_specs = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
    end
    loss = 0

    # We pull out the first sample from the loop in order to pass it back with the loss.
    n = 1
    i = bl.input_sample[n]
    dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), Tsit5(), saveat=bl.tsteps)
    dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_specs[n]), Tsit5(), saveat=bl.tsteps)
    loss += bl.output_metric(dd_sys, dd_spec)

    for n in 2:bl.N_samples
        i = bl.input_sample[n]
        dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), Tsit5(), saveat=bl.tsteps)
        dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_specs[n]), Tsit5(), saveat=bl.tsteps)
        loss += bl.output_metric(dd_sys, dd_spec)
        end
    loss, dd_sys, dd_spec
end

function individual_losses(bl::BAC_Problem, p)
    # Evalute the loss function of the BAC problem individually for each sample, this can be used for understanding the quality of the minimum found.

    @views begin
        p_sys = p[1:bl.dim_sys]
        p_specs = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
    end
    losses = zeros(bl.N_samples)

    # We pull out the first sample from the loop in order to pass it back with the loss.
    n = 1
    i = bl.input_sample[n]
    dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), Tsit5(), saveat=bl.tsteps)
    dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_specs[n]), Tsit5(), saveat=bl.tsteps)
    losses[n] = bl.output_metric(dd_sys, dd_spec)

    for n in 2:bl.N_samples
        i = bl.input_sample[n]
        dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), Tsit5(), saveat=bl.tsteps)
        dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_specs[n]), Tsit5(), saveat=bl.tsteps)
        losses[n] = bl.output_metric(dd_sys, dd_spec)
    end
    losses, dd_sys, dd_spec
end

export basic_bac_callback

function basic_bac_callback(p, loss, dd_sys, dd_spec)
    display(loss)
    # Tell sciml_train to not halt the optimization. If return true, then
    # optimization stops.
    return false
end


end # module
