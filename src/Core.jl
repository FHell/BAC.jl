import Base.@kwdef

export BAC_Loss
"""
Structure, containing a probabilistic behavioural control problem.


"""
@kwdef mutable struct BAC_Loss
    f_spec # f(dy, y, i, p, t)
    f_sys
    tsteps
    t_span
    input_sample # this is called with the sample as an argument and needs to return an input function i(t)
    output_metric
    N_samples::Int
    size_p_spec::Int
    size_p_sys::Int
    #dim_param::Int
    y0_spec
    y0_sys
    solver = Tsit5()
end

# A constructors for BAC_Loss with a default solver

function (bl::BAC_Loss)(p; solver_options...)
    # Evalute the loss function of the BAC problem
    
    p_sys = view(p,1:bl.size_p_sys)
    p_specs = [view(p,(bl.size_p_sys + 1 + (n - 1) * bl.size_p_spec):(bl.size_p_sys + n * bl.size_p_spec)) for n in 1:bl.N_samples]

    loss = 0.

    for n in 1:bl.N_samples
        i = bl.input_sample[n]
        loss += bl.output_metric(solve_sys_spec(bl, i, p_sys, p_specs[n]; solver_options...)...)
    end

    loss / bl.N_samples
end

function (bl::BAC_Loss)(n, p_sys, p_spec; solver_options...)
    # loss function for sample n only
    i = bl.input_sample[n]
    bl.output_metric(solve_sys_spec(bl, i, p_sys, p_spec; solver_options...)...)
end

export solve_sys_spec
"""
Solve for spec only.
Parameters:
- bl: existing bac_loss instance
- i: input to the system
- p_sys: parameters of the system
- p_spec: parameters of specification
"""
function solve_sys_spec(bl, i, p_sys, p_spec; solver_options...)
    dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), bl.solver; saveat=bl.tsteps, solver_options...)
    dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_spec), bl.solver; saveat=bl.tsteps, solver_options...)
    dd_sys, dd_spec
end

export solve_bl_n
"""
Solve PBC problem for a specific input.
Parameters:
- bl: existing bac_loss instance
- n: number of input in the array
- p: combined array of sys and spec parameters
"""
function solve_bl_n(bl::BAC_Loss, n::Int, p; solver_options...)
    p_sys = view(p,1:bl.size_p_sys)
    p_spec = view(p,(bl.size_p_sys + 1 + (n - 1) * bl.size_p_spec):(bl.size_p_sys + n * bl.size_p_spec))


    i = bl.input_sample[n]

    solve_sys_spec(bl, i, p_sys, p_spec; solver_options...)
end

export bac_spec_only
"""
Optimize spec parameters only. Can be used to check quality of the resulting minimizer and evaluate over-fitting. 
This function can be used to fit an already optimized system to a new sample of inputs.
Parameters:
- bl: BAC problem
- p_initial: array of parameters to begin optimization with
- optimizer: choose optimization algorithm (optinal)
""" 
function bac_spec_only(bl::BAC_Loss, p_initial; optimizer=DiffEqFlux.ADAM(0.01), optimizer_options=(:maxiters => 100,), solver_options...)
    # Optimize the specs only

    p = copy(p_initial)

    @views begin
        p_sys = p[1:bl.size_p_sys]
        p_specs = [p[bl.size_p_sys + 1 + (n - 1) * bl.size_p_spec:bl.size_p_sys + n * bl.size_p_spec] for n in 1:bl.N_samples]
    end

    for n in 1:bl.N_samples
        println(n)
        res = DiffEqFlux.sciml_train(
            x -> bl(n, p_sys, x; solver_options...), # Loss function for each n individually
            Array(p_specs[n]),
            optimizer;
            optimizer_options...
            )
        p_specs[n] .= res.minimizer
    end
    # println(p_sys)
    # println(p_specs)
    p
end

export matching_loss
"""
Evalute the loss function of the matching of the spec ODE to the system ODE.
This means for all input samples we take the same p_spec.
Parameters:
- bl: existing bac_loss instance
- p: combined array of sys and spec parameters
"""
function matching_loss(bl::BAC_Loss, p; solver_options...)
    # Evalute the loss function of the matching of the spec ODE to the system ODE
    # This means for all input samples we take the same P_spec

    @views begin
        p_sys = p[1:bl.size_p_sys]
        p_spec = p[bl.size_p_sys + 1:bl.size_p_sys + bl.size_p_spec]
    end

    loss = 0.

    for n in 1:bl.N_samples
        i = bl.input_sample[n]
        loss += bl.output_metric(solve_sys_spec(bl, i, p_sys, p_spec; solver_options...)...)
    end

    loss / bl.N_samples
end

export individual_losses
"""
Evaluate individual losses for each input.
Parameters:
- bl: existing bac_loss instance
- p: combined array of sys and spec parameters 
"""
function individual_losses(bl::BAC_Loss, p)
    # Return the array of losses
    @views begin
        p_sys = p[1:bl.size_p_sys]
        p_specs = [p[bl.size_p_sys + 1 + (n - 1) * bl.size_p_spec:bl.size_p_sys + n * bl.size_p_spec] for n in 1:bl.N_samples]
    end

    [bl(n, p_sys, p_specs[n]) for n in 1:bl.N_samples]
end


# Resampling the BAC_Loss
export resample
"""
Generate a new set of inputs.
Parameters:
- sampler: generator of inputs
- bac: PBC problem to be resampled
- n: number of samples. If not provided, bac.N_samples is used instead
"""
function resample(sampler, bac::BAC_Loss; n = 0)
    if n > 0
        bac.N_samples = n
    end
    new_input_sample = [sampler(n) for n in 1:bac.N_samples]
    BAC_Loss(bac.f_spec, bac.f_sys, bac.tsteps, bac.t_span, new_input_sample, bac.output_metric, bac.N_samples, bac.size_p_spec, bac.size_p_sys, bac.y0_spec, bac.y0_sys, bac.solver)
end

# Basic callback

export basic_bac_callback
"""
Quick callback function to be used in the sciml_train optimization process. Displays current loss.
"""
function basic_bac_callback(p, loss)
    display(loss)
    return false
end

function benchmark_callback(p, loss, tempt, templ, initial_time)
    if Base.Libc.time() - initial_time  >= 600
        return true
    else
        push!(templ, loss)
        display(loss)
# Tell sciml_train to not halt the optimization. If return true, then
# optimization stops.
        push!(tempt,Base.Libc.time())
        return false
    end
end

#=function diff(tempt,templ)
    t = zeros(length(tempt)-1)
    l = zeros(length(templ)-1)
    for i in 1:length(tempt)-1
        t[i] = tempt[i] - tempt[1]
        l[i] = templ[i]
    end
    return t, l
end=#
export confidence_interval
function confidence_interval(losses, delta)
    N = length(losses)+4
    d = (sum(losses[:].<delta)+2)/N
    c_high=d+(d*(1-d)/N)^0.5
    c_low=d-(d*(1-d)/N)^0.5
    return d, c_low, c_high
end
