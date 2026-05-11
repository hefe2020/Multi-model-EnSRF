using Random
using Distributions
using LinearAlgebra
using Statistics
using PyCall

include("helper_functions.jl")
import .HelperFunctions: shift, rk4singlestep, gaspari_cohn,
    local_matrix, model_step_ml_batch, ensemble_cov, diagonal_Q_basis,
    build_A_for_Q_basis, choose_obs_indices, build_H_from_obs_idx,
    forcing_vector, l96_rhs_forcing, add_diag_model_noise, apply_filter,
    estimate_diag_q_rank_deficient, forecast_lorenz_one_step,
    estimate_lambda_from_error_buffers, compute_ABC_lambda

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
import .plotting: global_single_forcing


function apply_da(filter_type, E, H, R, y, rho, rng)
    return apply_filter(filter_type, E, H, R, y, rho, rng;
        stochastic_enkf=stochastic_enkf,
        global_ensrf=global_ensrf,
        serial_ensrf=serial_ensrf)
end


function make_truth_and_obs(; N, dt, timesteps, obs_steps, obs_fraction,
                            obs_mode, std_obs, rng, F_true_vec)

    obs_idx = choose_obs_indices(N, obs_fraction; mode=obs_mode, rng=rng)
    H_obs = build_H_from_obs_idx(N, obs_idx)
    ny = length(obs_idx)

    R_all = (std_obs^2) .* Matrix{Float64}(I, ny, ny)

    x0 = randn(rng, N)
    for _ in 1:2000
        x0 = rk4singlestep((t, x) -> l96_rhs_forcing(x, F_true_vec), dt, 0.0, x0)
    end

    Y = zeros(Float64, N, timesteps + 1)
    Y[:, 1] = x0
    x = copy(x0)

    for k in 1:timesteps
        x = rk4singlestep((t, z) -> l96_rhs_forcing(z, F_true_vec), dt, (k - 1) * dt, x)
        Y[:, k + 1] = x
    end

    obs_idx_all = Int[]
    obs_all_list = Vector{Vector{Float64}}()
    obs_dist = MvNormal(zeros(ny), Symmetric(R_all))

    for k in 1:timesteps
        if k % obs_steps == 0
            push!(obs_idx_all, k)
            yk = H_obs * Y[:, k + 1] + rand(rng, obs_dist)
            push!(obs_all_list, yk)
        end
    end

    obs_all = hcat(obs_all_list...)

    return (
        Y = Y,
        obs_idx = obs_idx,
        H_obs = H_obs,
        R_all = R_all,
        obs_idx_all = obs_idx_all,
        obs_all = obs_all,
        ny = ny
    )
end


function run_q_calibration_single_forcing()
    N = 40
    dt = 0.05
    timesteps = 1000
    obs_steps = 5

    filter_type = "serial"
    Ne = 50

    std_obs = 0.5
    L_loc = 4.0
    rho = local_matrix(N, L_loc)

    alphaQ = 0.98
    rhoS = 0.98
    spread_frac = 0.10
    spread_frac_b = 0.05

    obs_fraction = 0.8
    obs_mode = "random"

    rng = MersenneTwister(42)

    F_true_vec  = forcing_vector(8.0, 8.0, 8.0, 8.0, N)
    F_wrong_vec = forcing_vector(7.4, 7.16, 6.4, 7.6, N)

    data = make_truth_and_obs(
        N=N, dt=dt, timesteps=timesteps, obs_steps=obs_steps,
        obs_fraction=obs_fraction, obs_mode=obs_mode,
        std_obs=std_obs, rng=rng, F_true_vec=F_true_vec
    )

    Y = data.Y
    H_obs = data.H_obs
    R_all = data.R_all
    R_diag = diag(R_all)
    obs_idx_all = data.obs_idx_all
    obs_all = data.obs_all
    ny = data.ny
    full_obs = (ny == N)

    println("Calibration obs_fraction = ", obs_fraction)
    println("Calibration ny           = ", ny)
    println("Calibration obs_idx      = ", data.obs_idx)

    Q_basis = diagonal_Q_basis(N)
    A_q = build_A_for_Q_basis(H_obs, Q_basis)

    signal_max = maximum(abs.(Y))
    sigma0 = spread_frac * signal_max
    sigmab = spread_frac_b * signal_max

    _ = nn._smodel.predict(zeros(Float32, 1, N, 1), verbose=0)

    X0 = reshape(Y[:, 1], :, 1) .+ sigmab

    Ens_lor = X0 .+ sigma0 .* randn(rng, N, Ne)
    Ens_ml  = X0 .+ sigma0 .* randn(rng, N, Ne)

    q_lor = fill(0.02, N)
    q_ml  = fill(0.02, N)

    Pf0_diag_lor = vec(var(Ens_lor, dims=2, corrected=true))
    Pf0_diag_ml  = vec(var(Ens_ml,  dims=2, corrected=true))

    if full_obs
        S_diag_lor = Pf0_diag_lor .+ q_lor .+ R_diag
        S_diag_ml  = Pf0_diag_ml  .+ q_ml  .+ R_diag
        S_lor = nothing
        S_ml = nothing
    else
        S_lor = H_obs * Diagonal(Pf0_diag_lor .+ q_lor) * H_obs' + R_all
        S_ml  = H_obs * Diagonal(Pf0_diag_ml  .+ q_ml ) * H_obs' + R_all
        S_diag_lor = nothing
        S_diag_ml = nothing
    end

    qhist_lor = zeros(Float64, N, timesteps + 1)
    qhist_ml  = zeros(Float64, N, timesteps + 1)

    qtrace_hist_lor = zeros(Float64, timesteps + 1)
    qtrace_hist_ml  = zeros(Float64, timesteps + 1)

    qhist_lor[:, 1] .= q_lor
    qhist_ml[:, 1]  .= q_ml
    qtrace_hist_lor[1] = sum(q_lor)
    qtrace_hist_ml[1]  = sum(q_ml)

    obs_counter = 1

    for k in 1:timesteps
        t = (k - 1) * dt

        E_lor = forecast_lorenz_one_step(Ens_lor, F_wrong_vec, dt, t)
        E_ml  = model_step_ml_batch(nn, Ens_ml)

        if k in obs_idx_all
            y_real_vec = obs_all[:, obs_counter]
            y_real = reshape(y_real_vec, :, 1)

            E_lor = add_diag_model_noise(E_lor, q_lor, rng)
            E_ml  = add_diag_model_noise(E_ml,  q_ml,  rng)

            xbar_f_lor, _, Pf_f_lor = ensemble_cov(E_lor)
            xbar_f_ml,  _, Pf_f_ml  = ensemble_cov(E_ml)

            Pf_diag_lor = diag(Pf_f_lor)
            Pf_diag_ml  = diag(Pf_f_ml)

            if full_obs
                d_lor = y_real_vec .- xbar_f_lor[:, 1]
                S_diag_lor = rhoS .* S_diag_lor .+ (1.0 - rhoS) .* (d_lor .^ 2)
                q_est_lor = max.(S_diag_lor .- Pf_diag_lor .- R_diag, 0.0)
                q_lor = alphaQ .* q_lor .+ (1.0 - alphaQ) .* q_est_lor

                d_ml = y_real_vec .- xbar_f_ml[:, 1]
                S_diag_ml = rhoS .* S_diag_ml .+ (1.0 - rhoS) .* (d_ml .^ 2)
                q_est_ml = max.(S_diag_ml .- Pf_diag_ml .- R_diag, 0.0)
                q_ml = alphaQ .* q_ml .+ (1.0 - alphaQ) .* q_est_ml
            else
                d_lor = y_real_vec .- (H_obs * xbar_f_lor[:, 1])
                S_lor = rhoS .* S_lor .+ (1.0 - rhoS) .* (d_lor * d_lor')
                C_lor = S_lor .- R_all .- (H_obs * Pf_f_lor * H_obs')
                C_lor = 0.5 .* (C_lor .+ C_lor')
                q_est_lor = estimate_diag_q_rank_deficient(C_lor, A_q; q_prev=q_lor, nonnegative=true)
                q_lor = max.(alphaQ .* q_lor .+ (1.0 - alphaQ) .* q_est_lor, 0.0)

                d_ml = y_real_vec .- (H_obs * xbar_f_ml[:, 1])
                S_ml = rhoS .* S_ml .+ (1.0 - rhoS) .* (d_ml * d_ml')
                C_ml = S_ml .- R_all .- (H_obs * Pf_f_ml * H_obs')
                C_ml = 0.5 .* (C_ml .+ C_ml')
                q_est_ml = estimate_diag_q_rank_deficient(C_ml, A_q; q_prev=q_ml, nonnegative=true)
                q_ml = max.(alphaQ .* q_ml .+ (1.0 - alphaQ) .* q_est_ml, 0.0)
            end

            Ens_lor = apply_da(filter_type, E_lor, H_obs, R_all, y_real, rho, rng)
            Ens_ml  = apply_da(filter_type, E_ml,  H_obs, R_all, y_real, rho, rng)

            obs_counter += 1
        else
            Ens_lor = E_lor
            Ens_ml  = E_ml
        end

        qhist_lor[:, k + 1] .= q_lor
        qhist_ml[:, k + 1]  .= q_ml

        qtrace_hist_lor[k + 1] = sum(q_lor)
        qtrace_hist_ml[k + 1]  = sum(q_ml)
    end

    return (
        qhist_lor = qhist_lor,
        qhist_ml = qhist_ml,
        qtrace_hist_lor = qtrace_hist_lor,
        qtrace_hist_ml = qtrace_hist_ml,
        F_true_vec = F_true_vec,
        F_wrong_vec = F_wrong_vec
    )
end


function learned_fixed_q(calib; burnin_frac=0.5)
    burnin = max(1, Int(round(burnin_frac * size(calib.qhist_lor, 2))))

    q_lor_fixed = vec(mean(calib.qhist_lor[:, burnin:end], dims=2))
    q_ml_fixed  = vec(mean(calib.qhist_ml[:,  burnin:end], dims=2))

    println("Learned fixed trace Q lor = ", round(sum(q_lor_fixed), digits=4))
    println("Learned fixed trace Q ml  = ", round(sum(q_ml_fixed), digits=4))

    return q_lor_fixed, q_ml_fixed
end


function run_weighted_hybrid_sweep_single_setup(q_lor_fixed, q_ml_fixed)
    N = 40
    dt = 0.05
    timesteps = 1000
    obs_steps = 5

    filter_type = "serial"
    Ne_lor = 50
    Ne_ml = 50
    Ne_bench = 50

    std_obs = 0.5
    L_loc = 4.0
    rho = local_matrix(N, L_loc)

    spread_frac = 0.10
    spread_frac_b = 0.05

    obs_fraction = 0.8
    obs_mode = "random"

    rng = MersenneTwister(42)

    do_adaptive_q = false
    do_branch_coupling_at_obs = true
    gamma_mm = 50.0

    λ_grid = collect(0.0:0.1:1.0)

    F_true_vec  = forcing_vector(8.0, 8.0, 8.0, 8.0, N)
    F_wrong_vec = forcing_vector(7.4, 7.16, 6.4, 7.6, N)

    data = make_truth_and_obs(
        N=N, dt=dt, timesteps=timesteps, obs_steps=obs_steps,
        obs_fraction=obs_fraction, obs_mode=obs_mode,
        std_obs=std_obs, rng=rng, F_true_vec=F_true_vec
    )

    Y = data.Y
    obs_idx = data.obs_idx
    H_obs = data.H_obs
    R_all = data.R_all
    obs_idx_all = data.obs_idx_all
    obs_all = data.obs_all
    ny = data.ny

    H_mm = Matrix{Float64}(I, N, N)

    println("Main obs_fraction = ", obs_fraction)
    println("Main ny           = ", ny)
    println("Main obs_idx      = ", obs_idx)

    signal_max = maximum(abs.(Y))
    sigma0 = spread_frac * signal_max
    sigmab = spread_frac_b * signal_max

    println("Highest absolute signal value = $(round(signal_max, digits=3))")
    println("Initial sigma0 = $(round(sigma0, digits=3))")
    println("Observation std = $(round(std_obs, digits=3))")

    _ = nn._smodel.predict(zeros(Float32, 1, N, 1), verbose=0)

    X0 = reshape(Y[:, 1], :, 1) .+ sigmab

    Ens_lor   = X0 .+ sigma0 .* randn(rng, N, Ne_lor)
    Ens_ml    = X0 .+ sigma0 .* randn(rng, N, Ne_ml)
    Ens_bench = X0 .+ sigma0 .* randn(rng, N, Ne_bench)

    q_lor = copy(q_lor_fixed)
    q_ml  = copy(q_ml_fixed)

    rmse_lor   = fill(NaN, timesteps + 1)
    rmse_ml    = fill(NaN, timesteps + 1)
    rmse_bench = fill(NaN, timesteps + 1)

    qtrace_hist_lor = zeros(Float64, timesteps + 1)
    qtrace_hist_ml  = zeros(Float64, timesteps + 1)

    analysis_times = Float64[]
    analysis_rmse_lor   = Float64[]
    analysis_rmse_ml    = Float64[]
    analysis_rmse_bench = Float64[]

    forecast_rmse_lor   = Float64[]
    forecast_rmse_ml    = Float64[]
    forecast_rmse_bench = Float64[]

    forecast_rmse_hybrid_by_λ = Dict(λ => Float64[] for λ in λ_grid)
    analysis_rmse_hybrid_by_λ = Dict(λ => Float64[] for λ in λ_grid)

    forecast_mean_hybrid_by_λ = Dict(λ => Vector{Vector{Float64}}() for λ in λ_grid)
    analysis_mean_hybrid_by_λ = Dict(λ => Vector{Vector{Float64}}() for λ in λ_grid)

    analysis_err_lor = Vector{Vector{Float64}}()
    analysis_err_ml = Vector{Vector{Float64}}()
    analysis_err_bench = Vector{Vector{Float64}}()

    forecast_err_lor = Vector{Vector{Float64}}()
    forecast_err_ml = Vector{Vector{Float64}}()
    forecast_err_bench = Vector{Vector{Float64}}()

    obs_counter = 1

    for k in 1:timesteps
        t = (k - 1) * dt

        E_lor   = forecast_lorenz_one_step(Ens_lor,   F_wrong_vec, dt, t)
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
            xbar_ml_f  = vec(mean(E_ml, dims=2))
            xbar_bench_f = vec(mean(E_bench, dims=2))

            push!(forecast_rmse_lor,   sqrt(mean((truth_now .- xbar_lor_f).^2)))
            push!(forecast_rmse_ml,    sqrt(mean((truth_now .- xbar_ml_f ).^2)))
            push!(forecast_rmse_bench, sqrt(mean((truth_now .- xbar_bench_f).^2)))

            push!(forecast_err_lor,   xbar_lor_f   .- truth_now)
            push!(forecast_err_ml,    xbar_ml_f    .- truth_now)
            push!(forecast_err_bench, xbar_bench_f .- truth_now)

            for λ in λ_grid
                xbar_hyb_f = (1.0 - λ) .* xbar_lor_f .+ λ .* xbar_ml_f
                push!(forecast_rmse_hybrid_by_λ[λ], sqrt(mean((truth_now .- xbar_hyb_f).^2)))
                push!(forecast_mean_hybrid_by_λ[λ], xbar_hyb_f)
            end

            if do_branch_coupling_at_obs
                R_mm = Symmetric(gamma_mm .* Pf_f_ml)
                E_lor = apply_da(filter_type, E_lor, H_mm, R_mm, reshape(xbar_ml_f, :, 1), rho, rng)
            end

            Ens_lor   = apply_da(filter_type, E_lor,   H_obs, R_all, y_real, rho, rng)
            Ens_ml    = apply_da(filter_type, E_ml,    H_obs, R_all, y_real, rho, rng)
            Ens_bench = apply_da(filter_type, E_bench, H_obs, R_all, y_real, rho, rng)

            xbar_lor_a = vec(mean(Ens_lor, dims=2))
            xbar_ml_a  = vec(mean(Ens_ml, dims=2))
            xbar_bench_a = vec(mean(Ens_bench, dims=2))

            push!(analysis_rmse_lor,   sqrt(mean((truth_now .- xbar_lor_a).^2)))
            push!(analysis_rmse_ml,    sqrt(mean((truth_now .- xbar_ml_a ).^2)))
            push!(analysis_rmse_bench, sqrt(mean((truth_now .- xbar_bench_a).^2)))

            push!(analysis_err_lor,   xbar_lor_a   .- truth_now)
            push!(analysis_err_ml,    xbar_ml_a    .- truth_now)
            push!(analysis_err_bench, xbar_bench_a .- truth_now)

            for λ in λ_grid
                xbar_hyb_a = (1.0 - λ) .* xbar_lor_a .+ λ .* xbar_ml_a
                push!(analysis_rmse_hybrid_by_λ[λ], sqrt(mean((truth_now .- xbar_hyb_a).^2)))
                push!(analysis_mean_hybrid_by_λ[λ], xbar_hyb_a)
            end

            push!(analysis_times, k * dt)
            obs_counter += 1
        else
            Ens_lor   = E_lor
            Ens_ml    = E_ml
            Ens_bench = E_bench
        end

        truth_now = Y[:, k + 1]

        rmse_lor[k + 1]   = sqrt(mean((truth_now .- vec(mean(Ens_lor, dims=2))) .^ 2))
        rmse_ml[k + 1]    = sqrt(mean((truth_now .- vec(mean(Ens_ml, dims=2))) .^ 2))
        rmse_bench[k + 1] = sqrt(mean((truth_now .- vec(mean(Ens_bench, dims=2))) .^ 2))

        qtrace_hist_lor[k + 1] = sum(q_lor)
        qtrace_hist_ml[k + 1]  = sum(q_ml)
    end

    return (
        dt = dt,
        timesteps = timesteps,
        λ_grid = λ_grid,
        Y = Y,
        obs_idx = obs_idx,
        obs_idx_all = obs_idx_all,
        obs_all = obs_all,
        F_true_vec = F_true_vec,
        F_wrong_vec = F_wrong_vec,

        do_adaptive_q = do_adaptive_q,
        do_branch_coupling_at_obs = do_branch_coupling_at_obs,

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

        forecast_rmse_hybrid_by_λ = forecast_rmse_hybrid_by_λ,
        analysis_rmse_hybrid_by_λ = analysis_rmse_hybrid_by_λ,

        forecast_mean_hybrid_by_λ = forecast_mean_hybrid_by_λ,
        analysis_mean_hybrid_by_λ = analysis_mean_hybrid_by_λ,

        analysis_err_lor_mat = hcat(analysis_err_lor...),
        analysis_err_ml_mat = hcat(analysis_err_ml...),
        analysis_err_bench_mat = hcat(analysis_err_bench...),

        forecast_err_lor_mat = hcat(forecast_err_lor...),
        forecast_err_ml_mat = hcat(forecast_err_ml...),
        forecast_err_bench_mat = hcat(forecast_err_bench...)
    )
end


calib = run_q_calibration_single_forcing()
q_lor_fixed, q_ml_fixed = learned_fixed_q(calib; burnin_frac=0.5)

results_all = run_weighted_hybrid_sweep_single_setup(q_lor_fixed, q_ml_fixed)

global_single_forcing(results_all);