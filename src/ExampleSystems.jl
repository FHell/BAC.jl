
module ExampleSystems

using ..BAC
using Random

function rand_fourier_input_generator(nn, N=10)
    a = randn(N)
    theta = 2*pi*rand(N)
    return t -> sum([a[n]*cos(n*t+theta[n]) for n in 1:N])
end

relu(x) = max(0., x)

using LightGraphs
struct nl_diff_dyn{T}
    L::T
end
function (dd::nl_diff_dyn)(dx, x, i, p, t)
    flows = dd.L * x
    @. dx = x - relu(p) * x^3 - flows
    dx[1] += i - x[1]
    nothing
end

struct StandardOutputMetric
    n_sys
    n_spec
end
function (som::StandardOutputMetric)(sol_sys, sol_spec)
    sum((sol_sys[som.n_sys, :] .- sol_spec[som.n_spec, :]) .^ 2)
end

function create_graph_example(dim_sys, av_deg, tsteps, N_samples)
    g_spec = SimpleGraph([Edge(1 => 2)])
    g_sys = barabasi_albert(dim_sys, av_deg)

    f_spec = nl_diff_dyn(laplacian_matrix(g_spec))
    f_sys = nl_diff_dyn(laplacian_matrix(g_sys))

    BAC.BAC_Problem(
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
    )
end

end # module