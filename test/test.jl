using Test
using Revise
using BAC

@info "Tests of BAC.jl" 

@testset "test of system initialization" begin
    dim_sys = 10
    bac_10 = BAC.create_graph_example(dim_sys, 3, 0.:0.1:10., 10)
end

