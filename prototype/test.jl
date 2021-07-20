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
nodefillc_sys = nodecolor[[1 for i in 1:10]]
nodefillc_spec = nodecolor[[1 for i in 1:2]]
nodefillc_sys[1] = nodecolor[2]
nodefillc_spec[1] = nodecolor[2]
nodefillc = nodecolor[[1,2,2,2,2,2,2,2,2,2]];
nodefillc_spec = nodecolor[[1,2]];
nodefillc_sys
ones(1,10)
using SparseArrays
using LightGraphs
spy(bac_10.f_sys)

L = bac_10.f_sys.L
L_spec = bac_10.f_spec.L

g1 = gplot(SimpleGraph(L), nodefillc=nodefillc_sys)
g2 = gplot(SimpleGraph(L_spec), nodefillc=nodefillc_spec)

savefig(g1, )
display(g1)
gplot(g1,g2)

using Cairo, Compose
# save to pdf
draw(PDF("karate.pdf", 16cm, 16cm), gplot(g1))
# save to png
draw(PNG("karate1.png", 16cm, 16cm), g1)
# save to svg
draw(SVG("karate.svg", 16cm, 16cm), gplot(g1))

relu(x) = max.(0., x)

using CairoMakie