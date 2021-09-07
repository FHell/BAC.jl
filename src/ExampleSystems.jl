function rand_fourier_input_generator(nn; N=10)
    a = randn(N)
    theta = 2*pi*rand(N)
    return t -> sum([a[n]*cos(n*t+theta[n]) for n in 1:N])
end

relu(x) = max.(0., x)

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

mutable struct NoTransientOutputMetric
    n_sys
    n_spec
    transient_time
end

function (som::NoTransientOutputMetric)(sol_sys, sol_spec)
    if sol_sys.retcode == :Success && sol_spec.retcode == :Success
        return sum((sol_sys[som.n_sys, som.transient_time:end] .- sol_spec[som.n_spec, som.transient_time:end]) .^ 2)
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
        zeros(dim_sys),
        Tsit5()
    )
end

mutable struct kuramoto_osc{w, N, K}
    w::w
    N::N
    K_av::K
end


function (dd::kuramoto_osc)(dx, x, i, p, t)
    # x -> Theta, p -> K
    K_total = 0.
    p_ind = 1
    for k in 1:dd.N
        dx[k] = x[k + dd.N]
    end
    for k in 1:dd.N
        for j in 1:(k-1)
            dx[k+dd.N] -= (relu(p[p_ind]) + 1.) * sin(x[k] - x[j])
            dx[j+dd.N] -= (relu(p[p_ind]) + 1.) * sin(x[j] - x[k])
            K_total += relu(p[p_ind]) + 1.
            p_ind += 1
        end
    end
    for k in 1:dd.N
      dx[k+dd.N] *= dd.K_av / K_total * (p_ind - 1) # (p_ind - 1) counts the number of edges
      dx[k+dd.N] += dd.w[k] - (relu(p[p_ind]) + 1.) * x[k+dd.N]
      p_ind += 1
    end
    # dx[1+dd.N] += dd.K_av * i
    dx[1+dd.N] += 1. * i
    nothing
end
## The specification, a kuramoto with inertia

function kuramoto_spec(dx, x, i, p, t)
    dx[1] = x[2]
    dx[2] = p[1] - (relu(p[2]) + 1.) * x[2]  + 1. * i# inertial node with a slackbus
    nothing
end


function kuramoto_out_metric(sol_sys, sol_spec; n_transient = length(0.:0.1:2pi))
    if sol_sys.retcode == :Success && sol_spec.retcode == :Success
        return sum((sol_sys[1, n_transient:end] .- sol_spec[1, n_transient:end]) .^ 2)
    else
        return Inf # Solvers failing is bad.
    end
end

function create_kuramoto_example(w, N_osc, dim_p_spec, K,  tsteps, N_samples; modes=5)
    f_sys = kuramoto_osc(w[1:N_osc], N_osc, K)
    BAC_Loss(
        kuramoto_spec,
        f_sys,
        tsteps,
        (tsteps[1], tsteps[end]),
        [rand_fourier_input_generator(n, N=modes) for n = 1:N_samples], # input function i(t) 
        kuramoto_out_metric, # phase at interface node
        N_samples,
        dim_p_spec, # Parameters in the spec
        N_osc * (N_osc - 1) ÷ 2 + N_osc, # Parameters in the sys
        zeros(2), # Initial state conditions for spec
        zeros(2 * N_osc), # Initial state conditions for sys
        Tsit5()
    )
end

