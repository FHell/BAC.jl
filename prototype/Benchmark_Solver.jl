cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Random
using OrdinaryDiffEq
using Plots
using LightGraphs
using Statistics
using Sundials
using ParameterizedFunctions, DiffEqDevTools, ODEInterfaceDiffEq

begin
    
struct nl_diff_dyn{T}
    L::T
end
relu(x) = max(0., x)
function (dd::nl_diff_dyn)(dx, x, i, p, t)
    flows = dd.L * x
    @. dx = x - relu(p) * x^3 - flows
    dx[1] += i - x[1]
    nothing
end
function rand_fourier_input_generator(nn, N=10)
    a = randn(N)
    theta = 2*pi*rand(N)
    return t -> sum([a[n]*cos(n*t+theta[n]) for n in 1:N])
end
# initialize for solving problem
dim_sys = 10;av_deg = 4;dim_spec = 2;N_sampODEInterfaceDiffEqles = 10;
g_sys = barabasi_albert(dim_sys, av_deg);f_sys = nl_diff_dyn(laplacian_matrix(g_sys))
y0_sys = zeros(dim_sys)
tsteps = 0.:0.1:10.;t_span = (tsteps[1], tsteps[end])
p_initial = ones(2*10+dim_sys);p_sys = p_initial[1:dim_sys]
#p_specs = [p_initial[dim_sys + 1 + (n - 1) * dim_spec:dim_sys + n * dim_spec] for n in 1:N_samples]
i = rand_fourier_input_generator(1)

# here i only compare different solver while solving sys_problem
prob = ODEProblem((dy,y,p,t) -> f_sys(dy, y,i(t), p, t),y0_sys,t_span,p_sys)
sol = solve(prob,Rodas5(),abstol=1/10^14,reltol=1/10^14)
test_sol = TestSolution(sol)
plot(sol)

end
## Low order RK methods
### High tolerances
# First we compare final errors of solutions with low order RK methods at high tolerances.

abstols = 1.0 ./ 10.0 .^ (4:7)
reltols = 1.0 ./ 10.0 .^ (1:4)

setups = [Dict(:alg=>BS3()),
          Dict(:alg=>Tsit5()),
          Dict(:alg=>RK4()),
          Dict(:alg=>DP5()),
          Dict(:alg=>OwrenZen3()),
          Dict(:alg=>OwrenZen4()),
          Dict(:alg=>OwrenZen5())]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:final)
plot(wp)

# Next we test interpolation errors:

abstols = 1.0 ./ 10.0 .^ (4:7)
reltols = 1.0 ./ 10.0 .^ (1:4)

setups = [Dict(:alg=>BS3()),
          Dict(:alg=>Tsit5()),
          Dict(:alg=>RK4()),
          Dict(:alg=>DP5()),
          Dict(:alg=>OwrenZen3()),
          Dict(:alg=>OwrenZen4()),
          Dict(:alg=>OwrenZen5())]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:L2)
plot(wp)

# Both interpolation tests and tests of final error show similar results. `BS3` does quite well, and `OwrenZen3`, `OwrenZen4`, `OwrenZen5`, and `RK4` achieve interpolation errors of about 1e-5.

### Low tolerances

# repeat the tests at low tolerances.

abstols = 1.0 ./ 10.0 .^ (8:11)
reltols = 1.0 ./ 10.0 .^ (5:8)

setups = [Dict(:alg=>BS3()),
          Dict(:alg=>Tsit5()),
          Dict(:alg=>RK4()),
          Dict(:alg=>DP5()),
          Dict(:alg=>OwrenZen3()),
          Dict(:alg=>OwrenZen4()),
          Dict(:alg=>OwrenZen5())]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:final)
plot(wp)

abstols = 1.0 ./ 10.0 .^ (8:11)
reltols = 1.0 ./ 10.0 .^ (5:8)

setups = [Dict(:alg=>BS3()),
          Dict(:alg=>Tsit5()),
          Dict(:alg=>RK4()),
          Dict(:alg=>DP5()),
          Dict(:alg=>OwrenZen3()),
          Dict(:alg=>OwrenZen4()),
          Dict(:alg=>OwrenZen5())]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:L2)
plot(wp)

# Out of the compared methods, `Tsit5`, `DP5`, and `OwrenZen5` seem to be the best methods for this problem at low tolerances, but also `OwrenZen4` performs similarly well. `OwrenZen5` and `OwrenZen4` can even achieve interpolation errors below 1e-9.


## Lazy interpolants
### High tolerances
# compare the Verner methods, which use lazy interpolants, at high tolerances. As reference include `OwrenZen4`.

abstols = 1.0 ./ 10.0 .^ (4:7)
reltols = 1.0 ./ 10.0 .^ (1:4)

setups = [Dict(:alg=>Vern6()),
          Dict(:alg=>Vern7()),
          Dict(:alg=>Vern8()),
          Dict(:alg=>Vern9()),
          Dict(:alg=>OwrenZen4())]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:final)
plot(wp)

abstols = 1.0 ./ 10.0 .^ (4:7)
reltols = 1.0 ./ 10.0 .^ (1:4)

setups = [Dict(:alg=>Vern6()),
          Dict(:alg=>Vern7()),
          Dict(:alg=>Vern8()),
          Dict(:alg=>Vern9()),
          Dict(:alg=>OwrenZen4())]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:L2)
plot(wp)

### Low tolerances
# repeat these tests and compare the Verner methods also at low tolerances.

abstols = 1.0 ./ 10.0 .^ (8:11)
reltols = 1.0 ./ 10.0 .^ (5:8)

setups = [Dict(:alg=>Vern6()),
          Dict(:alg=>Vern7()),
          Dict(:alg=>Vern8()),
          Dict(:alg=>Vern9()),
          Dict(:alg=>OwrenZen4())]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:final)
plot(wp)

abstols = 1.0 ./ 10.0 .^ (8:11)
reltols = 1.0 ./ 10.0 .^ (5:8)

setups = [Dict(:alg=>Vern6()),
          Dict(:alg=>Vern7()),
          Dict(:alg=>Vern8()),
          Dict(:alg=>Vern9()),
          Dict(:alg=>OwrenZen4())]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:L2)
plot(wp)

# It seems `Vern6` and `Vern7` are both well suited for the problem at low tolerances and outperform `OwrenZen4`, whereas at high tolerances `OwrenZen4` is more efficient.
##  compare of some common used solvers
abstols = 1.0 ./ 10.0 .^ (4:11)
reltols = 1.0 ./ 10.0 .^ (1:8)
setups = [Dict(:alg=>Tsit5()),
          Dict(:alg=>BS5()),
          Dict(:alg=>Vern7()),
          Dict(:alg=>Rosenbrock23()),
          Dict(:alg=>Rodas3()),
          Dict(:alg=>TRBDF2()),
          Dict(:alg=>CVODE_BDF()),
#          Dict(:alg=>rodas()), fail in deep compiling
#          Dict(:alg=>radau()), fail in deep compiling
          Dict(:alg=>RadauIIA5()),
          Dict(:alg=>ROS34PW1a()),
#          Dict(:alg=>lsoda()), fail to precompile LSODA
          ]

wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                      appxsol=test_sol,dense=false,
                      save_everystep=false,numruns=100,maxiters=10000000,
                      timeseries_errors=false,verbose=false)
plot(wp)
# BS5 and Tsit5 are the fastest.
