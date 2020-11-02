## In VSCode double hashes demarcate code cells that can be run with Shift-Enter

include("benchmark_setup.jl")

Random.seed!(42);

dim_sys = 10

bac_10 = create_graph_example(dim_sys, 3, 0.:0.1:10., 10)

# we can easily plot the input sample
plot(0:0.01:10, bac_10.input_sample, c=:gray, alpha=1, legend=false)

# a nonlinear diffusively coupled graph system.
# Specification is a two node version of the graph.
# => Can we make a large graph react like a small graph by tuning the non-linearity at the nodes?

p_initial = ones(2*10+dim_sys)

## Finished initialization
## Benchmarking

# Implement training with a set of optimizers
setups = [Dict(:opt=>DiffEqFlux.ADAM(0.01), :name=>"ADAM(0.01)"), 
Dict(:opt=>DiffEqFlux.ADAM(0.05), :name=>"ADAM(0.05)"), 
Dict(:opt=>DiffEqFlux.ADAM(0.1), :name=>"ADAM(0.1)"), 
Dict(:opt=>DiffEqFlux.ADAM(0.2), :name=>"ADAM(0.2)"), 
Dict(:opt=>DiffEqFlux.ADAM(0.3), :name=>"ADAM(0.3)"), 
Dict(:opt=>DiffEqFlux.ADAM(0.5), :name=>"ADAM(0.5)"), 
Dict(:opt=>DiffEqFlux.ADAM(0.7), :name=>"ADAM(0.7)"), 
          ]

t, l = train_set(1, setups, bac_10, p_initial) #Compiling everything
t, l = train_set(50, setups, bac_10, p_initial)

#tN, lN = single_train(3, DiffEqFlux.NewtonTrustRegion(), bac_100, relu.(res_100.minimizer))
#Above with error message TypeError: in typeassert, expected Float64, got a value of type ForwardDiff.Dual{Nothing,Float64,12}

# BFGS seems to get stuck with too larg maxiter or stepnorm
# tB, lB = single_train(20, DiffEqFlux.BFGS(initial_stepnorm = 0.01), bac_10, p_initial)

# Write training data into DataFrame
begin
    BenchResults = DataFrame(solver = String[], times = Float64[], loss = Float64[])
    for i in 1:length(setups)
        for j in 1:length(t[i])
            #BenchResults.solver = repeat(setups[i][:name], length(t[i]))
            push!(BenchResults.solver, setups[i][:name])
            push!(BenchResults.times, t[i][j])
            push!(BenchResults.loss, l[i][j])
        end
    end
    display(BenchResults)
end

CSV.write("$(@__FILE__).csv", BenchResults)

# Display of DataFrame grouped with different solver
for i in 1:length(setups)
    display(BenchResults[BenchResults.solver.==setups[i][:name],:])
end

plt = plot_bac_bench(BenchResults)
savefig(plt, "$(@__FILE__).png")