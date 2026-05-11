using Random
using Distributions
using LinearAlgebra
using Statistics
using PyCall
using Serialization


include("helper_functions.jl")
import .HelperFunctions: shift, rk4singlestep, gaspari_cohn,
    local_matrix, model_step_ml_batch, ensemble_cov, diagonal_Q_basis,
    build_A_for_Q_basis, forcing_vector, l96_rhs_forcing,
    forecast_lorenz_one_step, choose_obs_indices, build_H_from_obs_idx,
    estimate_diag_q_rank_deficient, add_diag_model_noise, apply_filter,
    compute_statewise_lambda, smooth_statewise_lambda,
    hybrid_mean_statewise

include("DA_schemes.jl")
import .DA_schemes: stochastic_enkf, global_ensrf, serial_ensrf

script_dir = @__DIR__
project_root = normpath(joinpath(script_dir, ".."))

sys = pyimport("sys")
pushfirst!(sys."path", script_dir)
pushfirst!(sys."path", project_root)

py_ml_model = pyimport("ml_model")
nn = py_ml_model.nn

include("plotting.jl")
import .plotting: plot_global_vs_statewise_across_forcings


function apply_da(filter_type, E, H, R, y, rho, rng)
    return apply_filter(filter_type, E, H, R, y, rho, rng;
        stochastic_enkf=stochastic_enkf,
        global_ensrf=global_ensrf,
        serial_ensrf=serial_ensrf)
end

results_global = deserialize("results_global.jls")


function run_statewise_main_for_forcing(; forcing_name, F_wrong_vec,
                                        q_lor_fixed, q_ml_fixed,
                                        shared_data, config)

    N = config.N
    dt = config.dt
    timesteps = config.timesteps
    filter_type = config.filter_type

    rho = config.rho
    rng = MersenneTwister(config.seed_main)

    F_true_vec = config.F_true_vec
    H_mm = Matrix{Float64}(I, N, N)

    Y = shared_data.Y
    obs_idx = shared_data.obs_idx
    H_obs = shared_data.H_obs
    R_all = shared_data.R_all
    obs_idx_all = shared_data.obs_idx_all
    obs_all = shared_data.obs_all

    signal_max = maximum(abs.(Y))
    sigma0 = config.spread_frac * signal_max
    sigmab = config.spread_frac_b * signal_max

    X0 = reshape(Y[:, 1], :, 1) .+ sigmab

    Ens_lor   = X0 .+ sigma0 .* randn(rng, N, config.Ne_lor)
    Ens_ml    = X0 .+ sigma0 .* randn(rng, N, config.Ne_ml)
    Ens_bench = X0 .+ sigma0 .* randn(rng, N, config.Ne_bench)

    q_lor = copy(q_lor_fixed)
    q_ml  = copy(q_ml_fixed)

    rmse_lor   = fill(NaN, timesteps + 1)
    rmse_ml    = fill(NaN, timesteps + 1)
    rmse_bench = fill(NaN, timesteps + 1)

    qtrace_hist_lor = zeros(Float64, timesteps + 1)
    qtrace_hist_ml  = zeros(Float64, timesteps + 1)

    analysis_times = Float64[]

    analysis_rmse_lor = Float64[]
    analysis_rmse_ml = Float64[]
    analysis_rmse_bench = Float64[]

    forecast_rmse_lor = Float64[]
    forecast_rmse_ml = Float64[]
    forecast_rmse_bench = Float64[]

    forecast_rmse_hybrid_statewise = Float64[]
    analysis_rmse_hybrid_statewise = Float64[]

    forecast_mean_hybrid_statewise = Vector{Vector{Float64}}()
    analysis_mean_hybrid_statewise = Vector{Vector{Float64}}()

    analysis_err_lor = Vector{Vector{Float64}}()
    analysis_err_ml = Vector{Vector{Float64}}()
    analysis_err_bench = Vector{Vector{Float64}}()

    forecast_err_lor = Vector{Vector{Float64}}()
    forecast_err_ml = Vector{Vector{Float64}}()
    forecast_err_bench = Vector{Vector{Float64}}()

    λ_state_hist = Vector{Vector{Float64}}()
    A_state_hist = Vector{Vector{Float64}}()
    B_state_hist = Vector{Vector{Float64}}()
    C_state_hist = Vector{Vector{Float64}}()

    err_lor_buf = Vector{Vector{Float64}}()
    err_ml_buf  = Vector{Vector{Float64}}()

    λ_state_current = fill(0.5, N)

    obs_counter = 1

    for k in 1:timesteps
        t = (k - 1) * dt

        E_lor   = forecast_lorenz_one_step(Ens_lor, F_wrong_vec, dt, t)
        E_ml    = model_step_ml_batch(nn, Ens_ml)
        E_bench = forecast_lorenz_one_step(Ens_bench, F_true_vec, dt, t)

        if k in obs_idx_all
            y_real_vec = obs_all[:, obs_counter]
            y_real = reshape(y_real_vec, :, 1)

            E_lor = add_diag_model_noise(E_lor, q_lor, rng)
            E_ml  = add_diag_model_noise(E_ml,  q_ml,  rng)

            _, _, Pf_f_ml = ensemble_cov(E_ml)

            truth_now = Y[:, k + 1]

            xbar_lor_f = vec(mean(E_lor, dims=2))
            xbar_ml_f = vec(mean(E_ml, dims=2))
            xbar_bench_f = vec(mean(E_bench, dims=2))

            push!(forecast_rmse_lor, sqrt(mean((truth_now .- xbar_lor_f).^2)))
            push!(forecast_rmse_ml, sqrt(mean((truth_now .- xbar_ml_f).^2)))
            push!(forecast_rmse_bench, sqrt(mean((truth_now .- xbar_bench_f).^2)))

            e_lor = xbar_lor_f .- truth_now
            e_ml = xbar_ml_f .- truth_now

            push!(forecast_err_lor, e_lor)
            push!(forecast_err_ml, e_ml)
            push!(forecast_err_bench, xbar_bench_f .- truth_now)

            push!(err_lor_buf, e_lor)
            push!(err_ml_buf, e_ml)

            if length(err_lor_buf) > config.λ_window
                popfirst!(err_lor_buf)
                popfirst!(err_ml_buf)
            end

            if length(err_lor_buf) >= config.λ_min_samples
                E_lor_buf = hcat(err_lor_buf...)
                E_ml_buf = hcat(err_ml_buf...)

                λ_raw, A_state, B_state, C_state =
                    compute_statewise_lambda(E_lor_buf, E_ml_buf)

                λ_raw = smooth_statewise_lambda(λ_raw; radius=config.λ_smooth_radius)

                λ_state_current =
                    config.λ_smooth_time .* λ_state_current .+
                    (1.0 - config.λ_smooth_time) .* λ_raw

                push!(A_state_hist, A_state)
                push!(B_state_hist, B_state)
                push!(C_state_hist, C_state)
            else
                push!(A_state_hist, fill(NaN, N))
                push!(B_state_hist, fill(NaN, N))
                push!(C_state_hist, fill(NaN, N))
            end

            push!(λ_state_hist, copy(λ_state_current))

            xbar_hyb_f = hybrid_mean_statewise(xbar_lor_f, xbar_ml_f, λ_state_current)
            push!(forecast_rmse_hybrid_statewise,
                  sqrt(mean((truth_now .- xbar_hyb_f).^2)))
            push!(forecast_mean_hybrid_statewise, xbar_hyb_f)

            if config.do_branch_coupling_at_obs
                R_mm = Symmetric(config.gamma_mm .* Pf_f_ml)
                E_lor = apply_da(
                    filter_type, E_lor, H_mm, R_mm,
                    reshape(xbar_ml_f, :, 1), rho, rng
                )
            end

            Ens_lor   = apply_da(filter_type, E_lor, H_obs, R_all, y_real, rho, rng)
            Ens_ml    = apply_da(filter_type, E_ml, H_obs, R_all, y_real, rho, rng)
            Ens_bench = apply_da(filter_type, E_bench, H_obs, R_all, y_real, rho, rng)

            xbar_lor_a = vec(mean(Ens_lor, dims=2))
            xbar_ml_a = vec(mean(Ens_ml, dims=2))
            xbar_bench_a = vec(mean(Ens_bench, dims=2))

            push!(analysis_rmse_lor, sqrt(mean((truth_now .- xbar_lor_a).^2)))
            push!(analysis_rmse_ml, sqrt(mean((truth_now .- xbar_ml_a).^2)))
            push!(analysis_rmse_bench, sqrt(mean((truth_now .- xbar_bench_a).^2)))

            push!(analysis_err_lor, xbar_lor_a .- truth_now)
            push!(analysis_err_ml, xbar_ml_a .- truth_now)
            push!(analysis_err_bench, xbar_bench_a .- truth_now)

            xbar_hyb_a = hybrid_mean_statewise(xbar_lor_a, xbar_ml_a, λ_state_current)
            push!(analysis_rmse_hybrid_statewise,
                  sqrt(mean((truth_now .- xbar_hyb_a).^2)))
            push!(analysis_mean_hybrid_statewise, xbar_hyb_a)

            push!(analysis_times, k * dt)
            obs_counter += 1
        else
            Ens_lor = E_lor
            Ens_ml = E_ml
            Ens_bench = E_bench
        end

        truth_now = Y[:, k + 1]

        rmse_lor[k + 1] =
            sqrt(mean((truth_now .- vec(mean(Ens_lor, dims=2))) .^ 2))
        rmse_ml[k + 1] =
            sqrt(mean((truth_now .- vec(mean(Ens_ml, dims=2))) .^ 2))
        rmse_bench[k + 1] =
            sqrt(mean((truth_now .- vec(mean(Ens_bench, dims=2))) .^ 2))

        qtrace_hist_lor[k + 1] = sum(q_lor)
        qtrace_hist_ml[k + 1] = sum(q_ml)
    end

    return (
        dt = dt,
        timesteps = timesteps,
        forcing_name = forcing_name,
        Y = Y,
        obs_idx = obs_idx,
        obs_idx_all = obs_idx_all,
        obs_all = obs_all,
        F_true_vec = F_true_vec,
        F_wrong_vec = F_wrong_vec,

        do_adaptive_q = false,
        do_branch_coupling_at_obs = config.do_branch_coupling_at_obs,

        fixed_q_lor_vec = q_lor_fixed,
        fixed_q_ml_vec = q_ml_fixed,
        fixed_q_lor_trace = sum(q_lor_fixed),
        fixed_q_ml_trace = sum(q_ml_fixed),

        rmse_lor = rmse_lor,
        rmse_ml = rmse_ml,
        rmse_bench = rmse_bench,

        qtrace_hist_lor = qtrace_hist_lor,
        qtrace_hist_ml = qtrace_hist_ml,
        q_lor = q_lor,
        q_ml = q_ml,

        analysis_times = analysis_times,
        analysis_rmse_lor = analysis_rmse_lor,
        analysis_rmse_ml = analysis_rmse_ml,
        analysis_rmse_bench = analysis_rmse_bench,

        forecast_rmse_lor = forecast_rmse_lor,
        forecast_rmse_ml = forecast_rmse_ml,
        forecast_rmse_bench = forecast_rmse_bench,

        forecast_rmse_hybrid_statewise = forecast_rmse_hybrid_statewise,
        analysis_rmse_hybrid_statewise = analysis_rmse_hybrid_statewise,

        forecast_mean_hybrid_statewise = forecast_mean_hybrid_statewise,
        analysis_mean_hybrid_statewise = analysis_mean_hybrid_statewise,

        analysis_err_lor_mat = hcat(analysis_err_lor...),
        analysis_err_ml_mat = hcat(analysis_err_ml...),
        analysis_err_bench_mat = hcat(analysis_err_bench...),

        forecast_err_lor_mat = hcat(forecast_err_lor...),
        forecast_err_ml_mat = hcat(forecast_err_ml...),
        forecast_err_bench_mat = hcat(forecast_err_bench...),

        λ_state_mat = hcat(λ_state_hist...),
        A_state_mat = hcat(A_state_hist...),
        B_state_mat = hcat(B_state_hist...),
        C_state_mat = hcat(C_state_hist...)
    )
end


function run_statewise_hybrid_forcing_sweep()
    N = 40
    dt = 0.05
    timesteps = 1000
    obs_steps = 5

    F_true_vec = forcing_vector(8.0, 8.0, 8.0, 8.0, N)

    config = (
        N = N,
        dt = dt,
        timesteps = timesteps,
        obs_steps = obs_steps,

        filter_type = "serial",
        Ne_calib = 50,
        Ne_lor = 50,
        Ne_ml = 50,
        Ne_bench = 50,

        std_obs = 0.5,
        rho = local_matrix(N, 4.0),

        alphaQ = 0.98,
        rhoS = 0.98,
        spread_frac = 0.10,
        spread_frac_b = 0.05,

        obs_fraction = 0.8,
        obs_mode = "random",

        seed_data = 42,
        seed_calib = 123,
        seed_main = 456,

        q_burnin_frac = 0.5,

        do_branch_coupling_at_obs = true,
        gamma_mm = 50.0,

        λ_min_samples = 5,
        λ_window = 20,
        λ_smooth_time = 0.8,
        λ_smooth_radius = 1,

        F_true_vec = F_true_vec
    )

    forcing_sweep = [
        ("F=(7.2, 6.0, 6.6, 7.0)",  forcing_vector(7.2, 6.0, 6.6, 7.0, N)),
        ("F=(7.2, 6.5, 6.6, 7.9)",  forcing_vector(7.2, 6.5, 6.6, 7.9, N)),
        ("F=(7.4, 7.16, 6.4, 7.6)", forcing_vector(7.4, 7.16, 6.4, 7.6, N)),
        ("F=(7.5, 7.0, 8.5, 8.0)",  forcing_vector(7.5, 7.0, 8.5, 8.0, N)),
        ("F=(6.4, 5.6, 4.8, 6.8)",  forcing_vector(6.4, 5.6, 4.8, 6.8, N))
    ]

    rng_data = MersenneTwister(config.seed_data)

    shared_data = make_truth_and_obs(
        N = N,
        dt = dt,
        timesteps = timesteps,
        obs_steps = obs_steps,
        obs_fraction = config.obs_fraction,
        obs_mode = config.obs_mode,
        std_obs = config.std_obs,
        rng = rng_data,
        F_true_vec = F_true_vec
    )

    println("obs_fraction = ", config.obs_fraction)
    println("ny           = ", shared_data.ny)
    println("obs_idx      = ", shared_data.obs_idx)

    _ = nn._smodel.predict(zeros(Float32, 1, N, 1), verbose=0)

    results_dict = Dict{String, Any}()
    q_calib_dict = Dict{String, Any}()

    for (forcing_name, F_wrong_vec) in forcing_sweep
        println("\nCalibrating Q for ", forcing_name)

        calib = calibrate_q_for_forcing(
            forcing_name = forcing_name,
            F_wrong_vec = F_wrong_vec,
            shared_data = shared_data,
            config = config
        )

        q_calib_dict[forcing_name] = calib

        println("Running statewise hybrid for ", forcing_name)

        results_dict[forcing_name] = run_statewise_main_for_forcing(
            forcing_name = forcing_name,
            F_wrong_vec = F_wrong_vec,
            q_lor_fixed = calib.q_lor_fixed,
            q_ml_fixed = calib.q_ml_fixed,
            shared_data = shared_data,
            config = config
        )
    end

    return (
        results = results_dict,
        q_calibration = q_calib_dict,
        config = config
    )
end


all_statewise_runs = run_statewise_hybrid_forcing_sweep()
results_statewise = all_statewise_runs.results

plot_global_vs_statewise_across_forcings(
    results_global,
    results_statewise;
    burnin = 5
)