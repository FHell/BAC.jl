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

x = [1 2; 3 4]
[1,2,3,4]
x[:,1]

a=1

length(a)

function test_m(bl, p, loss; input_sample = nothing, plot_options...)
      display(loss)
      colors_list = ["blue", "red", "green", "orange", "aqua", "coral", "darkorange", "deeppink", "cyan", "maroon"]
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
          plt = plot(layout = (length(input_sample),1))
          pl = [plot() for i in 1:length(input_sample)]
          for i in input_sample
                dd_sys, dd_spec = solve_bl_n(bl, i, p)
                plot!(plt, dd_sys, vars=1; label = "System output (sample $i)", color=colors_list[j], subplot = j, plot_options...)
                plot!(plt, dd_spec, vars=1; label = "Specification output (sample $i))", color=colors_list[j], linestyle = :dash, subplot = j, plot_options...)
                plot!(plt, dd_spec.t, bl.input_sample[i]; c=colors_list[j], alpha=0.8, label = "Input $i", linestyle = :dot, subplot = j, title = ("Input sample $i"), plot_options...)
                #title!("Input samples $i")
                j+=1
          end
          display(plt)
      end
    # Tell sciml_train to not halt the optimization. If return true, then
    # optimization stops.
      return false
end

test_m(bac_10, p_initial, l, input_sample=1:3)

input_sample=[1,2,3]
for i in input_sample
      print(i)
end

print(input_sample)
