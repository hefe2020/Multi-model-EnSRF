module plotting
export statewise_single_forcing, global_single_forcing, plot_global_vs_statewise_across_forcings, plot_weighted_hybrid_forcing_sweep

using Statistics, PyPlot
pygui(true)


function statewise_single_forcing(results_all_states)
    burnin = 5

    forecast_hyb_mean = mean(results_all_states.forecast_rmse_hybrid_statewise[burnin:end])
    analysis_hyb_mean = mean(results_all_states.analysis_rmse_hybrid_statewise[burnin:end])

    forecast_ml_mean = mean(results_all_states.forecast_rmse_ml[burnin:end])
    forecast_lor_mean = mean(results_all_states.forecast_rmse_lor[burnin:end])
    forecast_bench_mean = mean(results_all_states.forecast_rmse_bench[burnin:end])

    analysis_ml_mean = mean(results_all_states.analysis_rmse_ml[burnin:end])
    analysis_lor_mean = mean(results_all_states.analysis_rmse_lor[burnin:end])
    analysis_bench_mean = mean(results_all_states.analysis_rmse_bench[burnin:end])

    fig, ax = subplots(2, 1, figsize=(8, 8))

    ax[1].bar(
        ["Statewise Hybrid", "ML", "Lorenz", "Benchmark"],
        [forecast_hyb_mean, forecast_ml_mean, forecast_lor_mean, forecast_bench_mean]
    )
    ax[1].set_ylabel("Mean forecast RMSE")
    ax[1].set_title("Forecast RMSE comparison")
    ax[1].grid(true, axis="y")

    ax[2].bar(
        ["Statewise Hybrid", "ML", "Lorenz", "Benchmark"],
        [analysis_hyb_mean, analysis_ml_mean, analysis_lor_mean, analysis_bench_mean]
    )
    ax[2].set_ylabel("Mean analysis RMSE")
    ax[2].set_title("Analysis RMSE comparison")
    ax[2].grid(true, axis="y")

    tight_layout()
    show()

end



function global_single_forcing(results_all)

    λs = results_all.λ_grid
    forecast_means = [mean(results_all.forecast_rmse_hybrid_by_λ[λ][5:end]) for λ in λs]
    analysis_means = [mean(results_all.analysis_rmse_hybrid_by_λ[λ][5:end]) for λ in λs]

    figure()
    plot(λs, forecast_means, marker="o", label="Hybrid forecast")
    plot(λs, analysis_means, marker="s", label="Hybrid analysis")
    axhline(mean(results_all.forecast_rmse_ml[5:end]), linestyle="--", label="ML forecast")
    axhline(mean(results_all.forecast_rmse_lor[5:end]), linestyle="--", label="Lorenz forecast")
    axhline(mean(results_all.analysis_rmse_lor[5:end]), linestyle="--", label="Lorenz analysis")
    axhline(mean(results_all.analysis_rmse_ml[5:end]), linestyle="--", label="ML analysis")

    legend()
    grid(true)
    xlabel("λ (ML weight)")
    ylabel("Mean RMSE")
    title("Weighted hybrid sweep")
    show()
end



function plot_global_vs_statewise_across_forcings(results_global, results_statewise; burnin=5)
    forcing_names = sort(collect(keys(results_global)))
    ncases = length(forcing_names)

    forecast_ml_vals = Float64[]
    forecast_global_vals = Float64[]
    forecast_state_vals = Float64[]
    benchmark_forecast_vals = Float64[]

    analysis_ml_vals = Float64[]
    analysis_global_vals = Float64[]
    analysis_state_vals = Float64[]
    benchmark_analysis_vals = Float64[]

    best_λ_forecast_vals = Float64[]
    best_λ_analysis_vals = Float64[]

    for fname in forcing_names
        res_g = results_global[fname]
        res_s = results_statewise[fname]

        λs = res_g.λ_grid
        forecast_means_global = [mean(res_g.forecast_rmse_hybrid_by_λ[λ][burnin:end]) for λ in λs]
        analysis_means_global = [mean(res_g.analysis_rmse_hybrid_by_λ[λ][burnin:end]) for λ in λs]

        push!(forecast_global_vals, minimum(forecast_means_global))
        push!(analysis_global_vals, minimum(analysis_means_global))

        push!(best_λ_forecast_vals, λs[argmin(forecast_means_global)])
        push!(best_λ_analysis_vals, λs[argmin(analysis_means_global)])

        push!(forecast_ml_vals, mean(res_s.forecast_rmse_ml[burnin:end]))
        push!(analysis_ml_vals, mean(res_s.analysis_rmse_ml[burnin:end]))

        push!(forecast_state_vals, mean(res_s.forecast_rmse_hybrid_statewise[burnin:end]))
        push!(analysis_state_vals, mean(res_s.analysis_rmse_hybrid_statewise[burnin:end]))

        push!(benchmark_forecast_vals, mean(res_s.forecast_rmse_bench[burnin:end]))
        push!(benchmark_analysis_vals, mean(res_s.analysis_rmse_bench[burnin:end]))
    end

    x = collect(1:ncases)
    w = 0.2

    fig, ax = subplots(2, 1, figsize=(13, 8), sharex=true)

    b1 = ax[1].bar(x .- 1.5w, forecast_ml_vals,        width=w, label="Pure ML")
    b2 = ax[1].bar(x .- 0.5w, forecast_global_vals,    width=w, label="Best global λ")
    b3 = ax[1].bar(x .+ 0.5w, forecast_state_vals,     width=w, label="Statewise λ")
    b4 = ax[1].bar(x .+ 1.5w, benchmark_forecast_vals, width=w, label="Benchmark")
    ax[1].set_ylabel("Mean forecast RMSE")
    ax[1].set_title("Forecast: global vs statewise hybrid across forcing cases")
    ax[1].grid(true, axis="y")

    
    ax[2].bar(x .- 1.5w, analysis_ml_vals,        width=w, label="Pure ML")
    ax[2].bar(x .- 0.5w, analysis_global_vals,    width=w, label="Best global λ")
    ax[2].bar(x .+ 0.5w, analysis_state_vals,     width=w, label="Statewise λ")
    ax[2].bar(x .+ 1.5w, benchmark_analysis_vals, width=w, label="Benchmark")
    ax[2].set_ylabel("Mean analysis RMSE")
    ax[2].set_title("Analysis: global vs statewise hybrid across forcing cases")
    ax[2].grid(true, axis="y")

    ax[2].set_xticks(x)
    ax[2].set_xticklabels(
        ["$(fname)\nλf=$(round(best_λ_forecast_vals[i], digits=1)), λa=$(round(best_λ_analysis_vals[i], digits=1))"
         for (i, fname) in enumerate(forcing_names)],
        rotation=0
    )


    fig.legend(
        [b1, b2, b3, b4],
        ["Pure ML", "Best global λ", "Statewise λ", "Benchmark"],
        loc="lower center",
        ncol=4,
        frameon=false,
        bbox_to_anchor=(0.5, 0.03)
    )

    tight_layout(rect=[0, 0.08, 1, 1])
    show()
end



function plot_weighted_hybrid_forcing_sweep(results_all; burnin=5)
    forcing_names = sort(collect(keys(results_all)))
    ncases = length(forcing_names)

    fig, ax = subplots(2, ncases, figsize=(4*ncases, 8), sharex=true)

    if ncases == 1
        ax = reshape(ax, 2, 1)
    end

    for (j, fname) in enumerate(forcing_names)
        res = results_all[fname]
        λs = res.λ_grid

        forecast_means = [mean(res.forecast_rmse_hybrid_by_λ[λ][burnin:end]) for λ in λs]
        analysis_means = [mean(res.analysis_rmse_hybrid_by_λ[λ][burnin:end]) for λ in λs]

        forecast_ml = mean(res.forecast_rmse_ml[burnin:end])
        forecast_lor = mean(res.forecast_rmse_lor[burnin:end])

        analysis_ml = mean(res.analysis_rmse_ml[burnin:end])
        analysis_lor = mean(res.analysis_rmse_lor[burnin:end])

        forecast_bench = mean(res.forecast_rmse_bench[burnin:end])
        analysis_bench = mean(res.analysis_rmse_bench[burnin:end])

        λ_best_f = λs[argmin(forecast_means)]
        λ_best_a = λs[argmin(analysis_means)]

        ax[1, j].plot(λs, forecast_means, marker="o", label="Hybrid forecast")
        ax[1, j].axhline(forecast_ml, linestyle="--", label="ML")
        ax[1, j].axhline(forecast_lor, linestyle="--", label="Lorenz")
        ax[1, j].axhline(forecast_bench, linestyle="--", label="Benchmark")
        ax[1, j].set_title("$(fname)\nforecast best λ=$(round(λ_best_f, digits=2))")
        ax[1, j].grid(true)

        ax[2, j].plot(λs, analysis_means, marker="s", label="Hybrid analysis")
        ax[2, j].axhline(analysis_ml, linestyle="--", label="ML")
        ax[2, j].axhline(analysis_lor, linestyle="--", label="Lorenz")
        ax[2, j].axhline(analysis_bench, linestyle="--", color="red", label="Benchmark")
        ax[2, j].set_title("analysis best λ=$(round(λ_best_a, digits=2))")
        ax[2, j].grid(true)

        ax[2, j].set_xlabel("λ (ML weight)")
    end

    ax[1, 1].set_ylabel("Forecast RMSE")
    ax[2, 1].set_ylabel("Analysis RMSE")

    ax[1, 1].legend()
    ax[2, 1].legend()

    tight_layout()
    show()
end



end
