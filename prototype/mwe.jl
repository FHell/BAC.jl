cd(@__DIR__)
using Pkg
Pkg.activate(".")

using DiffEqFlux
using OrdinaryDiffEq

    
function l1(p; solver_options...)
    dd_spec = solve(ODEProblem((dy,y,p,t) -> dy .= p, zeros(2), (0., 10.), p), Tsit5(), saveat=0.:0.1:10., solver_options...)
    sum(dd_spec[end])
end

l1(ones(2)) # This works despite the missing semicolon

DiffEqFlux.sciml_train(
    l1,
    ones(2),
    DiffEqFlux.ADAM(0.5),
    maxiters = 2) # This doesn't with NamedTuple Error.

# I think what happens is that Julia doesn't require keyword arguments
# to be after the semicolon but it requires splatted keyword arguments
# to be after the semicolon. The following works:

function l2(p; solver_options...)
    dd_spec = solve(ODEProblem((dy,y,p,t) -> dy .= p, zeros(2), (0., 10.), p), Tsit5(), saveat=0.:0.1:10.; solver_options...)
    sum(dd_spec[end])
end

l2(ones(2))

DiffEqFlux.sciml_train(
    l2,
    ones(2),
    DiffEqFlux.ADAM(0.5),
    maxiters = 2)


function l3(p)
    ps = [p[1 + (n - 1) * 2:n * 2] for n in 1:10] # This is unused

    dd_spec = solve(ODEProblem((dy,y,p,t) -> dy .= p, zeros(2), (0., 10.), p[1:2]), Tsit5())
    sum(dd_spec[end])
end

l3(ones(20))

DiffEqFlux.sciml_train(
    l3,
    ones(20),
    DiffEqFlux.ADAM(0.5),
    maxiters = 2) # MethodError: no method matching iterate(::Nothing)
