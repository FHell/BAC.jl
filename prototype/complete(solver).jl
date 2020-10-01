cd(@__DIR__)
using Pkg
Pkg.activate(".")
# using Revise

# to prototype new features I copied everything into one file and made it mutable here.

using Random
using DiffEqFlux
using OrdinaryDiffEq
using Plots
using LightGraphs
using Statistics

begin #BAC.jl
    mutable struct BAC_Problem
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

    function (bl::BAC_Problem)(p; solver_options...)
          # Evalute the loss function of the BAC problem

          @views begin
              p_sys = p[1:bl.dim_sys]
              p_specs = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
          end

          loss = 0.

          # For plotting evaluate the first sample outside the loop:
          n = 1
          i = bl.input_sample[n]
          dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), bl.solver; saveat=bl.tsteps, solver_options...)
          dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_specs[n]), bl.solver; saveat=bl.tsteps, solver_options...)
          loss += bl.output_metric(dd_sys, dd_spec)

          for n in 2:bl.N_samples
              i = bl.input_sample[n]
              dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), bl.solver; saveat=bl.tsteps, solver_options...)
              dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_specs[n]), bl.solver; saveat=bl.tsteps, solver_options...)
              loss += bl.output_metric(dd_sys, dd_spec)
          end

          loss, dd_sys, dd_spec
      end


      function (bl::BAC_Problem)(n, p_sys, p_spec)
          i = bl.input_sample[n]
          dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), bl.solver; saveat=bl.tsteps, solver_options...)
          dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_spec), bl.solver; saveat=bl.tsteps, solver_options...)

          loss = bl.output_metric(dd_sys, dd_spec)

          loss, dd_sys, dd_spec
      end

      function bac_spec_only(bl::BAC_Problem, p_initial)
          # Optimize the specs only
          println("new2")

          p = copy(p_initial)
          @views begin
              p_sys = p[1:bl.dim_sys]
              p_specs = [p[bl.dim_sys + 1 + (n - 1) * bl.dim_spec:bl.dim_sys + n * bl.dim_spec] for n in 1:bl.N_samples]
          end


          for n in 1:bl.N_samples
            println(n)
            res = DiffEqFlux.sciml_train(
                x -> bl(n, p_sys, x),
                Array(p_specs[n]),
                DiffEqFlux.BFGS(initial_stepnorm = 0.01),
                maxiters = 100)
              p_specs[n] .= res.minimizer
          end

          p
      end

      function individual_loss(bl::BAC_Problem, p_sys, p_specs, n; solver_options...)
          # Evalute the individual loss contributed by the nth sample
          i = bl.input_sample[n]
          dd_sys = solve(ODEProblem((dy,y,p,t) -> bl.f_sys(dy, y, i(t), p, t), bl.y0_sys, bl.t_span, p_sys), bl.solver; saveat=bl.tsteps, solver_options...)
          dd_spec = solve(ODEProblem((dy,y,p,t) -> bl.f_spec(dy, y, i(t), p, t), bl.y0_spec, bl.t_span, p_specs[n]), bl.solver; saveat=bl.tsteps, solver_options...)
          bl.output_metric(dd_sys, dd_spec)
      end

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


      function basic_bac_callback(p, loss, dd_sys, dd_spec)
          display(loss)
          # Tell sciml_train to not halt the optimization. If return true, then
          # optimization stops.
          return false
      end
end

begin # ExampleSystems

    function rand_fourier_input_generator(nn, N=10)
        a = randn(N)
        theta = 2*pi*rand(N)
        return t -> sum([a[n]*cos(n*t+theta[n]) for n in 1:N])
    end

    relu(x) = max(0., x)

    struct nl_diff_dyn{T}
        L::T
    end
    function (dd::nl_diff_dyn)(dx, x, i, p, t)
        flows = dd.L * x
        @. dx = x - relu(p) * x^3 - flows
        dx[1] += i - x[1]
        nothing
    end

    mutable struct StandardOutputMetric
        n_sys
        n_spec
    end
    function (som::StandardOutputMetric)(sol_sys, sol_spec)
        sum((sol_sys[som.n_sys, :] .- sol_spec[som.n_spec, :]) .^ 2)
    end

    function create_graph_example(dim_sys, av_deg, tsteps, N_samples, slv)
        g_spec = SimpleGraph([Edge(1 => 2)])
        g_sys = barabasi_albert(dim_sys, av_deg)

        f_spec = nl_diff_dyn(laplacian_matrix(g_spec))
        f_sys = nl_diff_dyn(laplacian_matrix(g_sys))

        BAC_Problem(
            f_spec,
            f_sys,
            tsteps,
            (tsteps[1], tsteps[end]),
            [rand_fourier_input_generator(n) for n = 1:N_samples], # this is called with the sample as an argument and needs to return an input function i(t)
            StandardOutputMetric(1, 1),
            N_samples,
            2,
            dim_sys,
            zeros(2),
            zeros(dim_sys)
            ,solver = slv
        )
    end

end

begin # PlotUtils


    function plot_callback(p, loss, dd_sys, dd_spec)
        display(loss)
        plt = plot(dd_sys, vars=1)
        plot!(plt, dd_spec, vars=1)
        display(plt)
        # Tell sciml_train to not halt the optimization. If return true, then
        # optimization stops.
        return false
    end

    function l_heatmap(p_1a,p_1b,p_2a,p_2b, p, bac; sample_number = 1, axis_1 = 1, axis_2 = 2, stepsize = 1, title="")
        axis_m = zeros(2)
        axis = [axis_1,axis_2]
        for i in 1:2
            if axis[i]<=bac.dim_sys
                axis_m[i] = axis[i]
            else
                axis_m[i] = bac.dim_sys+bac.dim_spec*(sample_number-1)+axis[i]
            end
        end
        a_1 = Int(axis_m[1])
        a_2 = Int(axis_m[2])
        p_1s = (p_1a:stepsize:p_1b) .+ p[a_1]
        p_2s = (p_2a:stepsize:p_2b) .+ p[a_2]
        z = zeros(length(p_1s),length(p_2s))
        for i = 1:length(p_1s)
            for j = 1:length(p_2s)
                p_changed = copy(p)
                p_changed[a_1] = p_1s[i]
                p_changed[a_2] = p_2s[j]
                l, sol1, sol2 = bac(p_changed)
                z[i,j] = bac.output_metric(sol1, sol2)
            end
        end
        heatmap(p_1s, p_2s, z, aspect_ratio=1, xlabel="axis $axis_1", ylabel="axis $axis_2", title=title)
    end

    function display_heatmaps(p_1a, p_1b, p_2a, p_2b, p, bac; axis_1 = 1, axis_2 = 2, sample_number = 1,stepsize = 1, title = "")
        for s in sample_number
        for i in axis_1
            for j in axis_2
            if j in axis_1 && i in axis_2 && j > i || !(j in axis_1) || !(i in axis_2)
                display(l_heatmap(p_1a, p_1b, p_2a, p_2b, p, bac; axis_1 = i, axis_2 = j,sample_number = s,title = title))
            end
            end
        end
        end
    end

end # module

Random.seed!(42);

dim_sys = 10

bac_10 = create_graph_example(dim_sys, 3, 0.:0.1:10., 10, Rosenbrock23())

bac_100 = create_graph_example(dim_sys, 3, 0.:0.1:10., 100, Rosenbrock23())
# a nonlinear diffusively coupled graph system.
# Specification is a two node version of the graph.
# => Can we make a large graph react like a small graph by tuning the non-linearity at the nodes?

p_initial = ones(2*10+dim_sys)

# bac_1 implements the loss function. We are looking for parameters that minimize it, it can be evaluated
# directly on a parameter array:
l, sol1, sol2 = bac_10(p_initial)

# Plot callback plots the solutions passed to it:
plot_callback(p_initial, l, sol1, sol2)

# Underlying the loss function is the output metric comparing the two trajectories:
bac_10.output_metric(sol1, sol2)

# Train with 10 samples and relatively large ADAM step size: (1.5 minutes on my Laptop)
res_10 = DiffEqFlux.sciml_train(
    bac_10,
    p_initial,
    DiffEqFlux.ADAM(0.5),
    maxiters = 200,
    cb = basic_bac_callback
    )

# We can check the quality of the resulting minimizer by optimizing the specs only (a much simpler repeated 2d optimization problem)
p2 = bac_spec_only(bac_10, res_10.minimizer)
losses = individual_losses(bac_10, p2)
median(losses)

# In order to understand how much we were overfitting with respect to the concrete sample, we resample
# That is, we generate a new problem with a new sample from the same distribution:
bac_10_rs = resample(rand_fourier_input_generator, bac_10)
p3 = bac_spec_only(bac_10_rs, res_10.minimizer)
losses_rs = individual_losses(bac_10_rs, p3)
median(losses_rs)
# This will give us information on the system tuning with a sample different from the one that the tuning was optimized for.

# We warmed up the optimization with a very small number of samples,
# we can now initialize a higher sampled optimization using the system parameters found in the lower one:

p_100 = ones(2*100+dim_sys)
p_100[1:dim_sys] .= res_10.minimizer[1:dim_sys]

# Optimizing only the specs is a task linear in the number of samples,
# the idea is that this will help with warming up the optimization
# We can also study the quality of the tuning found by the optimization based on a small number of Samples
p_100_initial = bac_spec_only(bac_100, p_100)
losses_100_initial = individual_losses(bac_100, p_100_initial)
# Todo: Indication of bug or not completely understood behaviour!!
median(losses_100_initial) # This is much larger (factor 5-10) than the losses_10_rs version. It shouldn't be. Needs to be investigated!!!!!
# Possibility: THe optimization in bac_spec_only is not doing its job very well, switch to ADAM?

# Train the full system:
res_100 = DiffEqFlux.sciml_train(
    bac_100,
    p_100_initial,
    DiffEqFlux.ADAM(0.1),
    # DiffEqFlux.BFGS(initial_stepnorm = 0.01),
    maxiters = 5,
    cb = basic_bac_callback
    )

# Continue improving it for 150 Steps with some plotting in between:
for i in 1:30
    global res_100
    res_100 = DiffEqFlux.sciml_train(
        bac_100,
        relu.(res_100.minimizer),
        DiffEqFlux.ADAM(0.1),
        # DiffEqFlux.BFGS(initial_stepnorm = 0.01),
        maxiters = 5,
        cb = basic_bac_callback
        )
    l, sol1, sol2 = bac_100(res_100.minimizer);
    plot_callback(res_100.minimizer, l, sol1, sol2)
end
