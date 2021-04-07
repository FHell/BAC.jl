global x = rand(1000)
function sum_arg(x)
    s = 0.0
    for i in x
        s += i
        println(Base.Libc.time())
    end
    return s
end;
x = ones(1000)
Mytime1 = @timed sum_arg(x)
Mytime1.time

Mytime2 = @elapsed sum_arg(x)

Mytime2

using GraphPlot

nodecolor = [colorant"orange", colorant"lightseagreen"]
nodefillc = nodecolor[[1,2,2,2,2,2,2,2,2,2]];
nodefillc_spec = nodecolor[[1,2]];

using SparseArrays
using LightGraphs
spy(bac_10.f_sys)

L = bac_10.f_sys.L
L_spec = bac_10.f_spec.L

g1 = gplot(SimpleGraph(L), nodefillc=nodefillc)
g2 = gplot(SimpleGraph(L_spec), nodefillc=nodefillc_spec)
gplot(g1,g2)
