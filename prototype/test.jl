mutable struct Foo
           x::Float64
           y::Float64

           Foo(;x::Float64, y::Float64) = new(x, y)
           Foo() = new()
            end

Foo()

mutable struct BAC
      f_spec::Int
      f_sys::Int
      solver::String
end

struct BAC
      fields
end


global x = rand(1000)
function sum_arg(x)
      s = 0.0
      for i in x
            s += i
            println(Base.Libc.time())
      end
      return s
      end;
x = ones(1000)
Mytime1 = @timed sum_arg(x)
Mytime1.time

Mytime2 = @elapsed sum_arg(x)

Mytime2

res_10 = DiffEqFlux.sciml_train(
    p -> bac_10(p; abstol=1e-2, reltol=1e-2),
    p_initial,
    DiffEqFlux.ADAM(0.5),
    maxiters = 5,
    #cb = basic_bac_callback
    cb = (p, l) -> plot_callback(bac_10, p, l)
    )


dd_sys_test, dd_spec_test = solve_bl_n(bac_10, 1, p_initial)
plt = plot(dd_sys_test, vars=1; label = "System output")
plot!(plt, dd_spec_test, vars=1; label = "Specification output")
plot!(plt, dd_spec_test.t, bac_10.input_sample[1]; c=:gray, alpha=0.75, label = "Input")

display(plt)

plot(dd_sys_test.t,dd_sys_test'[:,1].+1.)

dd_sys_test'

function plot_callback_offset(bl, p, loss; loss_array = nothing, input_sample = nothing, fig_name = nothing, plot_options...)
    display(loss)
    colors_list = ["blue", "red", "green", "orange", "blue1"]
    plt = plot()
    isnothing(input_sample) ? input_sample = rand(1:bl.N_samples) : nothing
    if length(input_sample) == 1
          dd_sys, dd_spec = solve_bl_n(bl, input_sample[1], p)
          plt = plot(dd_sys, vars=1; label = "System output", plot_options...)
          plot!(plt, dd_spec, vars=1; label = "Specification output", plot_options...)
          plot!(plt, dd_spec.t, bl.input_sample[input_sample[1]]; c=:gray, alpha=0.75, label = "Input", plot_options...)
          title!("Input sample $(input_sample[1])")
          display(plt)
    else
        j = 1
        for i in input_sample
              dd_sys, dd_spec = solve_bl_n(bl, i, p)
              plot!(dd_sys.t, dd_sys[1,:].+2*(j-1), vars=1; label = "System output (sample $i)", color=colors_list[j], yaxis = nothing, plot_options...)
              plot!(plt, dd_spec.t, dd_spec[1,:].+2*(j-1), vars=1; label = "Specification output (sample $i))", color=colors_list[j], linestyle = :dash, plot_options...)
              #plot!(plt, dd_spec.t, bl.input_sample[i]; c=colors_list[j], alpha=0.8, label = "Input $i", linestyle = :dot, plot_options...)
              j+=1
        end
        title!("Input samples $(input_sample)")
        display(plt)
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

test_palette = [palette(:tab20)[Int(floor((i+1)/2))] for i in 1:2*length(palette(:tab20))]

losses

sum(losses[:].>0.1)/length(losses)

prob_dist(bac_100, res_100.minimizer, 0.3)

confidence_interval(bac_100, res_100.minimizer, 0.3)

function confidence_interval_sort(losses, delta)
    N = length(losses)+4
    sort!(losses)
    d = (sum(losses[:].<delta)+2)/N
    c_high=d+(d*(1-d)/N)^0.5
    c_low=d-(d*(1-d)/N)^0.5
    return d, c_low, c_high
end

g_sys = barabasi_albert(10, 3)

using GraphPlot

nodecolor = [colorant"orange", colorant"lightseagreen"]
nodefillc = nodecolor[[1,2,2,2,2,2,2,2,2,2]]

gplot(g_sys,nodefillc = nodefillc)

function plot_callback_test(bl, p, loss; loss_array = nothing, input_sample = nothing, fig_name = nothing, plot_options...)
    display(loss)
    offset = 2
    plt = plot()
    isnothing(input_sample) ? input_sample = rand(1:bl.N_samples) : nothing
    if length(input_sample) == 1
          dd_sys, dd_spec = solve_bl_n(bl, input_sample[1], p)
          plt = plot(dd_sys, vars=1; label = "System output", plot_options...)
          plot!(plt, dd_spec, vars=1; label = "Specification output", plot_options...)
          plot!(plt, dd_spec.t, bl.input_sample[input_sample[1]]; c=:gray, alpha=0.75, label = "Input", plot_options...)
          title!("Input sample $(input_sample[1])")
          display(plt)
    else
        j = 1
        for i in input_sample
              dd_sys, dd_spec = solve_bl_n(bl, i, p)
              plot!(dd_sys.t, dd_sys[1,:].+offset*(j-1), vars=1; label = false, color_palette = :tab20, plot_options...) #yaxis = nothing "System output (sample $i)"
              plot!(plt, dd_spec.t, dd_spec[1,:].+offset*(j-1), vars=1; label = false, linestyle = :dash, plot_options...)
              # "Specification output (sample $i))"
              j+=1
        end
        plot!([0.,0.01],[0.,0.];label = "System output", c=:gray)
        plot!([0.,0.01],[0.,0.];label = "Specification output", c=:gray, linestyle = :dash)
        #plot!(label = "System output", c=:gray)
        #plot!(label = "Specification output", c=:gray, linestyle = :dash)

        samples_line = ""
        for s in @view samples[1:end-1]
            samples_line*="$s, "
        end
        samples_line*="$(samples[end])"
        title!("Samples "*samples_line)
        display(plt)
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

plot(sin)
plot!([0.,0.01],[0.,0.];label = "test")

plot_callback_test(bac_100, p_100_initial, l, input_sample = samples)

dd_sys, dd_spec = solve_bl_n(bac_100, 1, res_100.minimizer)

plot(dd_sys)

plot(dd_spec)

gplot(bac_10.f_sys)
