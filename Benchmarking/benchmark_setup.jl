## In VSCode double hashes demarcate code cells that can be run with Shift-Enter

# For exploring features in the package here is a non package include based environment for running the code.
cd(@__DIR__)
using Pkg
Pkg.activate(".")

using Random
using DiffEqFlux
using OrdinaryDiffEq
using Plots
using LightGraphs
using Statistics
using DataFrames, Pipe, CSV

include("../src/Core.jl")
include("../src/ExampleSystems.jl")
include("../src/PlotUtils.jl")
include("../src/Benchmark.jl")
include("benchmark_make_plots.jl")
