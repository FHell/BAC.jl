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
    if sol_sys.retcode == :Success && sol_spec.retcode == :Success
        return sum((sol_sys[som.n_sys, :] .- sol_spec[som.n_spec, :]) .^ 2)
    else
        return Inf # Solvers failing is bad.
    end
end

function create_graph_example(dim_sys, av_deg, tsteps, N_samples)
    g_spec = SimpleGraph([Edge(1 => 2)])
    g_sys = barabasi_albert(dim_sys, av_deg)

    f_spec = nl_diff_dyn(laplacian_matrix(g_spec))
    f_sys = nl_diff_dyn(laplacian_matrix(g_sys))

    BAC_Loss(
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
