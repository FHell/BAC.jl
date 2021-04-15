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
Base.getindex(lp::LossPrecision,i::Int) = lp.times[i]
Base.getindex(lp::LossPrecision,::Colon) = lp.times
Base.getindex(lp_set::LossPrecisionSet,i::Int) = lp_set.lps[i]
Base.getindex(lp_set::LossPrecisionSet,::Colon) = lp_set.lps

# Plotreceipt for benchmark of solvers
@recipe function f(lp::LossPrecision)
  seriestype --> :path
  label -->  lp.name
  linewidth --> 3
  yguide --> "Time (s)"
  xguide --> "Error"
  xscale --> :log10
  yscale --> :log10
  marker --> :auto
  lp.errors,lp.times
end

@recipe function f(lp_set::LossPrecisionSet)
  seriestype --> :path
  linewidth --> 3
  yguide --> "Time (s)"
  xguide --> "Error"
  xscale --> :log10
  yscale --> :log10
  marker --> :auto
  errors = Vector{Any}(undef,0)
  times = Vector{Any}(undef,0)
  for i in 1:length(lp_set)
    push!(errors,lp_set[i].errors)
    push!(times,lp_set[i].times)
  end
  label -->  reshape(lp_set.names,1,length(lp_set))
  errors,times
end


## Training functions
function single_train(n_max, optimizer, bac_loss, p_initial)
    initial_time = Base.Libc.time()
    tempt = [initial_time, ]
    templ = [bac_loss(p_initial), ]
    DiffEqFlux.sciml_train(
        bac_loss,
        p_initial,
        optimizer,
        maxiters = n_max,
        cb = (p ,l) -> benchmark_callback(p, l, tempt, templ, initial_time)
        )
    tempt .-= initial_time
    return tempt, templ
end

function train_set(n_max, setups, bac_loss, p_initial)
    N = length(setups)
    t = Vector{Vector{Float64}}(undef,N)
    l = Vector{Vector{Float64}}(undef,N)
    for i in 1:N
        t[i], l[i] = single_train(n_max, setups[i][:opt], bac_loss, p_initial)
    end
    return t, l
end
