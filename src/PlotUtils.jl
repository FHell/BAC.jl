function plot_callback(bl, p, loss; plot_options...)
    display(loss)
    dd_sys, dd_spec = solve_bl_n(bl, 1, p)
    plt = plot(dd_sys, vars=1; label = "System output", plot_options...)
    plot!(plt, dd_spec, vars=1; label = "Specification output", plot_options...)
    display(plt)
    # Tell sciml_train to not halt the optimization. If return true, then
    # optimization stops.
    return plt
end

function plot_callback_save(bl, p, loss, fig_name; plot_options...)
    png(plot_callback(bl, p, loss; plot_options...), fig_name)
    return false
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
