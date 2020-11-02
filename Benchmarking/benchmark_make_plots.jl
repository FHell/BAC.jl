
# Plot of time-loss figure
function plot_bac_bench(BenchResults, setups)
    tempt = filter(:solver => ==(setups[1][:name]), BenchResults)
    plt = scatter(tempt.times, tempt.loss,
            title = "Training loss value",
            label = setups[1][:name])
    for i in 2:length(setups)
        tempt = filter(:solver => ==(setups[i][:name]), BenchResults)
        plt = scatter!(tempt.times, tempt.loss,
                label = setups[i][:name])

    end
    plt
end

# StatsPlots support DataFrames with a marco @df for easy ploting in different solver.
using StatsPlots


function plot_bac_bench(BenchResults)
    @df BenchResults scatter(
    :times,
    :loss,
    group = :solver,
    title = "Training loss value",
    m = (0.8, [:+ :h :star7], 7),
    bg = RGB(0.2, 0.2, 0.2))
end
##
