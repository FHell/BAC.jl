export plot_callback
"""
Plot trajectories of the first nodes of sys and spec, and display the loss.
Parameters:
- bl: BAC problem
- p: array of sys and spec parameters, usually res.minimizer
- loss: value of the output distance
Optional parameters:
- loss_array: array to append value of loss to. Can be used to plot progress of the optimization over iterations.
- scenario_nums: number(s) of scenarios to be plotted. If not provided, a scenario is chosen randomly.
- fig_name: path and file name to save the plot. If not provided, plot is not saved.
- offset: defines by how much each scenario is offset from the previous one for better plot readability. If set to -1, all samples are plotted on subplots instead of a single figure.
- plot_options: various plot options
"""
function plot_callback(bl, p, loss; loss_array=nothing, scenario_nums=nothing, fig_name=nothing, offset=2, dim = 1, plot_options...)
    display(loss)
    plt = plot()
    isnothing(scenario_nums) ? scenario_nums = rand(1:bl.N_samples) : nothing
    if length(scenario_nums) == 1
        dd_sys, dd_spec = solve_bl_n(bl, scenario_nums[1], p, dim = dim)
        plt = plot(dd_sys, vars=1; label="System output", plot_options...)
        plot!(plt, dd_spec, vars=1; label="Specification output", plot_options...)
        plot!(plt, dd_spec.t, bl.inputs[scenario_nums[1]]; c=:gray, alpha=0.75, label="Input", plot_options...)
        title!("Input scenario $(scenario_nums[1])")
        display(plt)
    else
        j = 1
        if offset == -1
            plt = plot(layout = (length(scenario_nums),1))
            for i in scenario_nums
                dd_sys, dd_spec = solve_bl_n(bl, i, p, dim = dim)
                plot!(plt, dd_sys, vars=1; label = "System output (scenario $i)", c = palette(:tab20)[2*i-1], subplot = j, plot_options...)
                plot!(plt, dd_spec, vars=1; label = "Specification output (scenario $i))", c = palette(:tab20)[2*i], linestyle = :dash, subplot = j, plot_options...)
                plot!(plt, dd_spec.t, bl.inputs[i]; c=:gray, alpha=0.5, label = "Input $i", subplot = j, title = ("Input scenario $i"), plot_options...)
                #title!("Input scenarios $i")
                j+=1
          end
          xlabel!("t")
          ylabel!("output")
          display(plt)
        else
            for i in scenario_nums
                dd_sys, dd_spec = solve_bl_n(bl, i, p)
                plot!(dd_sys.t, dd_sys[1,:] .+ offset * (j - 1), vars=1; label=false, color_palette=:tab20, plot_options...) # yaxis = nothing "System output (scnenario $i)"
                plot!(plt, dd_spec.t, dd_spec[1,:] .+ offset * (j - 1), vars=1; label=false, linestyle=:dash, plot_options...) # "Specification output (scnenario $i))"
                j += 1
            end
            plot!([0.,0.01], [0.,0.];label="System output", c=:gray)
            plot!([0.,0.01], [0.,0.];label="Specification output", c=:gray, linestyle=:dash)
            scenarios_line = ""
            for s in @view scenario_nums[1:end - 1]
                scenarios_line *= "$s, "
            end
            scenarios_line *= "$(scenario_nums[end])"
            title!("Scenarios " * scenarios_line)
            xlabel!("t")
            ylabel!("output")
            display(plt)
        end
      
    end
    if !isnothing(fig_name)
          savefig(fig_name)
    end
    if !isnothing(loss_array)
          append!(loss_array, loss)
    end
      # Tell sciml_train to not halt the optimization.
      # If return true, then optimization stops.
    return false
end

export plot_sys_graph
"""
Plot graph representations of optimized system and spec system.
"""
function plot_sys_graph(bl::BAC_Loss)
    nodecolor = [colorant"orange", colorant"lightseagreen"]
    nodefillc_sys = nodecolor[[1 for i in 1:bl.dim_sys]];
    nodefillc_spec = nodecolor[[1 for i in 1:bl.dim_spec]];
    nodefillc_sys[1] = nodecolor[2]
    nodefillc_spec[1] = nodecolor[2]
    L_sys = bl.f_sys.L
    L_spec = bl.f_spec.L
    g1 = gplot(SimpleGraph(L_sys), nodefillc=nodefillc_sys)
    g2 = gplot(SimpleGraph(L_spec), nodefillc=nodefillc_spec)
    display(g1)
    display(g2)
end

function l_heatmap(p_1a,p_1b,p_2a,p_2b, p, bac; sample_number = 1, axis_1 = 1, axis_2 = 2, stepsize = 1, title="")
    axis_m = [axis_1,axis_2]
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

function loss_heatmap(p_1a,p_1b,p_2a,p_2b, p, bac; axis_1 = 1, axis_2 = 2, stepsize = 1, title="")
axis_m = [axis_1,axis_2]
z = zeros(length(p_1a:stepsize:p_1b),length(p_2a:stepsize:p_2b))

for n in 1:bac.N_samples
    axis_m[1] = axis_1
    axis_m[2] = bac.dim_sys+bac.dim_spec*(n-1)+axis_2

    a_1 = Int(axis_m[1])
    a_2 = Int(axis_m[2])
    global p_1s = (p_1a:stepsize:p_1b) .+ p[a_1]
    global p_2s = (p_2a:stepsize:p_2b) .+ p[a_2]

    for i = 1:length(p_1s)
        for j = 1:length(p_2s)
            p_changed = copy(p)
            p_changed[a_1] = p_1s[i]
            p_changed[a_2] = p_2s[j]
            l, sol1, sol2 = bac(p_changed)
            z[i,j] += bac.output_metric(sol1, sol2)
        end
    end
end

    heatmap(p_1s, p_2s, z, aspect_ratio=1, xlabel="axis $axis_1", ylabel="axis $axis_2", title=title)
end
