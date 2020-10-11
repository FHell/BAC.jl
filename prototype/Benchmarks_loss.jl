# For exploring features in the package here is a non package include based environment for running the code.
cd(@__DIR__)
using Pkg
Pkg.activate(".")
# Pkg.instantiate()

using Random
using DiffEqFlux
using OrdinaryDiffEq
using Plots
using LightGraphs
using Statistics
using ParameterizedFunctions, DiffEqDevTools

include("../src/Core.jl")
include("../src/ExampleSystems.jl")
include("../src/PlotUtils.jl")

Random.seed!(42);

dim_sys = 10

bac_10 = create_graph_example(dim_sys, 3, 0.:0.1:10., 10)

# a nonlinear diffusively coupled graph system.
# Specification is a two node version of the graph.
# => Can we make a large graph react like a small graph by tuning the non-linearity at the nodes?

p_initial = ones(2*10+dim_sys)

# bac_1 implements the loss function. We are looking for parameters that minimize it, it can be evaluated
# directly on a parameter array:
l = bac_10(p_initial) # 110
l = bac_10(p_initial; abstol=1e-2, reltol=1e-2) # 108

sol1, sol2 = solve_bl_n(bac_10, 1, p_initial)
#error_estimate=:final
#sol1.k # parameters
prob = sol1.prob
sol = solve(prob,Rodas5(),abstol=1/10^14,reltol=1/10^14)
test_sol = TestSolution(sol) # TestSolution(interp::DESolution) = TestSolution(nothing,nothing,interp,true)
#plot(sol)
#plot(test_sol.interp)
sol .== test_sol.interp
sol_row = solve(prob,Tsit5(),abstol=1/10^1,reltol=1/10^1)
#sol_row.interp
#plot(sol.t,sol_row)
errorsol=appxtrue(sol_row, test_sol)
errorsol.errors[:final] #0.0106...

abstols = 1.0 ./ 10.0 .^ (8:11)
reltols = 1.0 ./ 10.0 .^ (5:8)

setups = [Dict(:alg=>BS3()),
          Dict(:alg=>Tsit5()),]
wp = WorkPrecisionSet(prob,abstols,reltols,setups;
                      appxsol=test_sol,maxiters=Int(1e5),error_estimate=:final)
wp = WorkPrecision(prob, Tsit5(), abstols, reltols;  appxsol=test_sol,maxiters=Int(1e5),error_estimate=:final)
plot(wp)

sol1.errors[error_estimate]
plot(sol1)
p_rands = [rand(30) .+ 0.5 for i in 1:20]
@time ls_2 = [bac_10(p; abstol=1e-2, reltol=1e-2) for p in p_rands] # 0.178718 seconds (753.29 k allocations: 59.235 MiB)
@time ls_22 = [bac_10(p_rands[end]; abstol=1e-2, reltol=1e-2)]
@time ls_3 = [bac_10(p; abstol=1e-3, reltol=1e-3) for p in p_rands] # 0.273185 seconds (839.18 k allocations: 65.494 MiB, 26.01% gc time)
@time ls_4 = [bac_10(p; abstol=1e-4, reltol=1e-4) for p in p_rands] # 0.244564 seconds (1.06 M allocations: 80.751 MiB)
@time ls_5 = [bac_10(p; abstol=1e-5, reltol=1e-5) for p in p_rands] # 0.328403 seconds (1.44 M allocations: 107.804 MiB, 12.27% gc time)
@time ls_10 = [bac_10(p; abstol=1e-10, reltol=1e-10) for p in p_rands] # 1.535774 seconds (10.63 M allocations: 750.371 MiB, 12.09% gc time)
@time ls_14 = [bac_10(p; abstol=1e-14, reltol=1e-14) for p in p_rands] # 8.515796 seconds (65.10 M allocations: 4.452 GiB, 11.33% gc time)

bac_10.solver=TRBDF2()
@time sls_2 = [bac_10(p; abstol=1e-2, reltol=1e-2) for p in p_rands] # 0.496420 seconds (894.82 k allocations: 67.331 MiB, 12.33% gc time)
@time sls_3 = [bac_10(p; abstol=1e-3, reltol=1e-3) for p in p_rands] # 0.543552 seconds (1.45 M allocations: 104.120 MiB)
@time sls_4 = [bac_10(p; abstol=1e-4, reltol=1e-4) for p in p_rands] # 0.818758 seconds (2.16 M allocations: 153.118 MiB, 6.94% gc time)
@time sls_5 = [bac_10(p; abstol=1e-5, reltol=1e-5) for p in p_rands] # 1.100352 seconds (3.28 M allocations: 230.709 MiB, 7.91% gc time)
@time sls_6 = [bac_10(p; abstol=1e-10, reltol=1e-10) for p in p_rands] # 10.811276 seconds (49.19 M allocations: 3.282 GiB, 6.53% gc time)

println(mean(sls_5 .- sls_6))
println(mean(ls_6 .- sls_6))
println(mean(ls_5 .- sls_6))

using Zygote
gradient(p -> bac_10(p; abstol=1e-3, reltol=1e-3), p_rands[1])
@time gradient(p -> bac_10(p; abstol=1e-3, reltol=1e-3), p_rands[1])

mutable struct LossPrecision
  loss
  abstols
  reltols
  errors
  times
  name
  N::Int
end

mutable struct LossPrecisionSet
  lps::Vector{LossPrecision}
  N::Int
  abstols
  reltols
  setups
  names
  numruns
end

function LossPrecision(ps,opt,abstols,reltols;
                       name=nothing,appxsol=nothing,numruns=20,seconds=2,kwargs...)
  N = length(abstols)
  errors = Vector{Float64}(undef,N)
  times = Vector{Float64}(undef,N)
  ls = Vector{Float64}(undef,length(ps))
  if name === nothing
    name = "LP-Alg"
  end
    for i in 1:N
      bac_10.solver = opt
        ls = [bac_10(p;kwargs...,abstol=abstols[i],reltol=reltols[i])  for p in ps]


      errors[i] = mean(ls.- appxsol)

      benchmark_f = let ps=ps,abstols=abstols,reltols=reltols,kwargs=kwargs
        bac_10.solver = opt
          () -> @elapsed [bac_10(p;
                               abstol = abstols[i],
                               reltol = reltols[i], kwargs...)  for p in ps]
      end
      benchmark_f() # pre-compile

      b_t = benchmark_f()
      if b_t > seconds
        times[i] = b_t
      else
        times[i] = mapreduce(i -> benchmark_f(), min, 2:numruns; init = b_t)
      end
    end
  return LossPrecision(ls,abstols,reltols,errors,times,name,N)
end

function LossPrecisionSet(ps,
                          abstols,reltols,setups;
                          print_names=false,names=nothing,appxsol=nothing,
                          test_dt=nothing,kwargs...)
  N = length(setups)
  @assert names === nothing || length(setups) == length(names)
  lps = Vector{LossPrecision}(undef,N)
  if names === nothing
    names = [string(nameof(typeof(setup[:alg]))) for setup in setups]
  end
  for i in 1:N
    println(setups[i][:alg])
    print_names && println(names[i])
      lps[i] = LossPrecision(ps,setups[i][:alg],abstols,reltols;
                                 appxsol=appxsol,
                                 name=names[i],kwargs...)
    println(lps[i])
  end
  return LossPrecisionSet(lps,N,abstols,reltols,setups,names,nothing)
end

function Base.show(io::IO, lp::LossPrecision)
  println(io,"Name: $(lp.name)")
  println(io,"Times: $(lp.times)")
  println(io,"Errors: $(lp.errors)")
end
Base.length(lp::LossPrecision) = lp.N
Base.length(lp_set::LossPrecisionSet) = lp_set.N
p_rands = [rand(30) .+ 0.5 for i in 1:20]
#length(p_rands)
abstols = 1.0 ./ 10.0 .^ (2:4)
reltols = 1.0 ./ 10.0 .^ (2:4)
test_ls = [bac_10(p; abstol=1e-14, reltol=1e-14) for p in p_rands]
lp = LossPrecision(p_rands, Tsit5(), abstols, reltols; appxsol=test_ls,maxiters=Int(1e5))
lpp = LossPrecision(p_rands, TRBDF2(), abstols, reltols; appxsol=test_ls,maxiters=Int(1e5))
plot(lp.errors,lp.times)
plot(lpp.errors,lpp.times)
scatter!(lpp.errors,lpp.times)

lpp.errors
wp.errors
wp.
plot(wp.prob)

show(wp)
show(lp)
length(wp)

length(lpp)
wp
lpp
plot(wp)

setups = [Dict(:alg=>BS3()),
          Dict(:alg=>Tsit5()),
          Dict(:alg=>RK4()),
          Dict(:alg=>DP5()),
          Dict(:alg=>TRBDF2()),
          Dict(:alg=>OwrenZen3()),
          Dict(:alg=>OwrenZen4()),
          Dict(:alg=>OwrenZen5())]

#length(setups)
lps = LossPrecisionSet(p_rands, abstols, reltols, setups;appxsol=test_ls)

plot(lps)
