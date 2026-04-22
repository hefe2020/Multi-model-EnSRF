module HelperFunctions

export shift, rk4singlestep, gaspari_cohn, local_matrix, model_step_ml_batch,
       ensemble_cov, add_diag_model_noise, apply_filter, forcing_vector,
       l96_rhs_forcing, forecast_lorenz_one_step, choose_obs_indices,
       build_H_from_obs_idx, diagonal_Q_basis, build_A_for_Q_basis,
       estimate_diag_q_rank_deficient, estimate_lambda_from_error_buffers, 
       compute_ABC_lambda, statewise_error_stats, statewise_lambda, 
       compute_statewise_lambda, smooth_statewise_lambda, hybrid_mean_statewise



using Statistics
using LinearAlgebra
using Random
using Distributions


function shift(x::AbstractVector, n::Integer)
    return circshift(x, -n)
end

function rk4singlestep(fun::Function, dt::Float64, t0::Float64, y0::AbstractVector{Float64})
    f1 = fun(t0, y0)
    f2 = fun(t0 + dt / 2, y0 .+ (dt / 2) .* f1)
    f3 = fun(t0 + dt / 2, y0 .+ (dt / 2) .* f2)
    f4 = fun(t0 + dt, y0 .+ dt .* f3)
    return y0 .+ dt .* (f1 .+ 2 .* f2 .+ 2 .* f3 .+ f4) ./ 6.0
end

function gaspari_cohn(r::Real, L::Real)
    rr = abs(r) / L
    if rr >= 2.0
        return 0.0
    elseif rr <= 1.0
        return 1.0 - (5.0 / 3.0) * rr^2 + (5.0 / 8.0) * rr^3 + 0.5 * rr^4 - 0.25 * rr^5
    else
        return 4.0 -
               5.0 * rr +
               (5.0 / 3.0) * rr^2 +
               (5.0 / 8.0) * rr^3 -
               0.5 * rr^4 +
               (1.0 / 12.0) * rr^5 -
               (2.0 / (3.0 * rr))
    end
end

function local_matrix(Nx::Int, L::Real)
    rho = zeros(Float64, Nx, Nx)
    for i in 1:Nx
        for j in 1:Nx
            dist = min(abs(i - j), Nx - abs(i - j))
            rho[i, j] = gaspari_cohn(dist, L)
        end
    end
    return rho
end

function model_step_ml_batch(E::AbstractMatrix{Float64})
    N, Ne = size(E)
    Xin = reshape(Float32.(permutedims(E)), Ne, N, 1)
    py_out = nn._smodel.predict(Xin, verbose=0)
    Xout = Array(py_out)[:, :, 1]
    return Float64.(permutedims(Xout))
end

function ensemble_cov(E::AbstractMatrix{Float64})
    xbar = mean(E, dims=2)
    A = E .- xbar
    Pf = (A * A') / (size(E, 2) - 1)
    return xbar, A, Pf
end

function add_diag_model_noise(E::AbstractMatrix{Float64},
                              qdiag::AbstractVector{Float64},
                              rng::AbstractRNG)
    std = sqrt.(max.(qdiag, 0.0))
    return E .+ reshape(std, :, 1) .* randn(rng, size(E)...)
end

function apply_filter(filter_type::String,
                      E::AbstractMatrix{Float64},
                      H::AbstractMatrix{Float64},
                      R::AbstractMatrix{Float64},
                      y::AbstractMatrix{Float64},
                      rho::AbstractMatrix{Float64},
                      rng::AbstractRNG)
    if filter_type == "stochastic"
        return stochastic_enkf(E, H, R, y; rho=rho, rng=rng, inflation=1.0)
    elseif filter_type == "serial"
        return serial_ensrf(E, H, R, y, rho)
    else
        return global_ensrf(E, H, R, y, rho, 1.0)
    end
end


function forcing_vector(F1::Float64, F2::Float64, F3::Float64, F4::Float64, N::Int)
    F = zeros(Float64, N)
    for i in 1:N
        if i <= 10
            F[i] = F1
        elseif i <= 20
            F[i] = F2
        elseif i <= 30
            F[i] = F3
        else
            F[i] = F4
        end
    end
    return F
end

function l96_rhs_forcing(x::AbstractVector{Float64}, Fvec::AbstractVector{Float64})
    return (shift(x, 1) .- shift(x, -2)) .* shift(x, -1) .- x .+ Fvec
end

function forecast_lorenz_one_step(E::AbstractMatrix{Float64},
                                  Fvec::AbstractVector{Float64},
                                  dt::Float64,
                                  t::Float64)
    _, Ne = size(E)
    E_out = similar(E)

    for i in 1:Ne
        E_out[:, i] = rk4singlestep(
            (tt, x) -> l96_rhs_forcing(x, Fvec),
            dt,
            t,
            E[:, i]
        )
    end

    return E_out
end


function choose_obs_indices(N::Int, obs_fraction::Float64; mode::String="regular", rng=Random.default_rng())
    ny = max(1, round(Int, obs_fraction * N))

    if mode == "regular"
        idx = round.(Int, range(1, N, length=ny))
        idx = unique(clamp.(idx, 1, N))
        while length(idx) < ny
            missing = setdiff(1:N, idx)
            push!(idx, first(missing))
            sort!(idx)
        end
        return idx[1:ny]
    elseif mode == "random"
        return sort(randperm(rng, N)[1:ny])
    else
        error("Unknown obs mode: $mode")
    end
end

function build_H_from_obs_idx(N::Int, obs_idx::AbstractVector{Int})
    ny = length(obs_idx)
    H = zeros(Float64, ny, N)
    for (i, j) in enumerate(obs_idx)
        H[i, j] = 1.0
    end
    return H
end


function diagonal_Q_basis(N::Int)
    Q_basis = Vector{Matrix{Float64}}(undef, N)
    for i in 1:N
        Qi = zeros(Float64, N, N)
        Qi[i, i] = 1.0
        Q_basis[i] = Qi
    end
    return Q_basis
end

function build_A_for_Q_basis(H::AbstractMatrix{Float64}, Q_basis::Vector{Matrix{Float64}})
    ny = size(H, 1)
    A_q = zeros(Float64, ny^2, length(Q_basis))
    for p in 1:length(Q_basis)
        A_q[:, p] = vec(H * Q_basis[p] * H')
    end
    return A_q
end

function estimate_diag_q_rank_deficient(C::AbstractMatrix{Float64},
                                        A_q::AbstractMatrix{Float64};
                                        q_prev::Union{Nothing,AbstractVector{Float64}}=nothing,
                                        nonnegative::Bool=true)
    b = vec(C)
    q_est = A_q \ b

    if q_prev !== nothing
        bad = .!isfinite.(q_est)
        q_est[bad] .= q_prev[bad]
    end

    q_est = real.(q_est)

    if nonnegative
        q_est = max.(q_est, 0.0)
    end

    return q_est
end

function estimate_lambda_from_error_buffers(err_lor_buf::Vector{Vector{Float64}},
                                            err_ml_buf::Vector{Vector{Float64}})
    T = min(length(err_lor_buf), length(err_ml_buf))

    if T == 0
        return 0.5, NaN, NaN, NaN
    end

    A = mean(dot(err_lor_buf[t], err_lor_buf[t]) for t in 1:T)
    B = mean(dot(err_ml_buf[t],  err_ml_buf[t])  for t in 1:T)
    C = mean(dot(err_lor_buf[t], err_ml_buf[t]) for t in 1:T)

    denom = A + B - 2C
    if !isfinite(denom) || abs(denom) < 1e-12
        return 0.5, A, B, C
    end

    λ = (A - C) / denom
    λ = clamp(λ, 0.0, 1.0)

    return λ, A, B, C
end

function compute_ABC_lambda(results_all; burnin=5)
    E_lor = results_all.forecast_err_lor_mat[:, burnin:end]
    E_ml  = results_all.forecast_err_ml_mat[:, burnin:end]

    T = size(E_lor, 2)

    A = mean(dot(E_lor[:, t], E_lor[:, t]) for t in 1:T)
    B = mean(dot(E_ml[:, t],  E_ml[:, t])  for t in 1:T)
    C = mean(dot(E_lor[:, t], E_ml[:, t]) for t in 1:T)

    denom = A + B - 2C
    λ_star = abs(denom) < 1e-12 ? NaN : clamp((A - C) / denom, 0.0, 1.0)

    return (A=A, B=B, C=C, λ_star=λ_star)
end


function statewise_error_stats(results_all; burnin=5)
    E_lor = results_all.forecast_err_lor_mat[:, burnin:end]
    E_ml  = results_all.forecast_err_ml_mat[:, burnin:end]

    rmse_lor_state = sqrt.(mean(E_lor.^2, dims=2))[:]
    rmse_ml_state  = sqrt.(mean(E_ml.^2,  dims=2))[:]
    diff_state = rmse_lor_state .- rmse_ml_state

    return (
        rmse_lor_state = rmse_lor_state,
        rmse_ml_state = rmse_ml_state,
        diff_state = diff_state
    )
end

function statewise_lambda(results_all; burnin=5)
    E_lor = results_all.forecast_err_lor_mat[:, burnin:end]
    E_ml  = results_all.forecast_err_ml_mat[:, burnin:end]

    N = size(E_lor, 1)
    λ_state = zeros(N)

    for i in 1:N
        a = E_lor[i, :]
        b = E_ml[i, :]

        A = mean(a.^2)
        B = mean(b.^2)
        C = mean(a .* b)

        denom = A + B - 2C
        λ_state[i] = abs(denom) < 1e-12 ? NaN : clamp((A - C) / denom, 0.0, 1.0)
    end

    return λ_state
end

function compute_statewise_lambda(E_lor_err::AbstractMatrix, E_ml_err::AbstractMatrix)
    N, T = size(E_lor_err)
    λ_state = zeros(Float64, N)
    A_state = zeros(Float64, N)
    B_state = zeros(Float64, N)
    C_state = zeros(Float64, N)

    for i in 1:N
        a = view(E_lor_err, i, :)
        b = view(E_ml_err,  i, :)

        A = mean(a .^ 2)
        B = mean(b .^ 2)
        C = mean(a .* b)

        denom = A + B - 2C
        λ = abs(denom) < 1e-12 ? 0.5 : clamp((A - C) / denom, 0.0, 1.0)

        λ_state[i] = λ
        A_state[i] = A
        B_state[i] = B
        C_state[i] = C
    end

    return λ_state, A_state, B_state, C_state
end

function smooth_statewise_lambda(λ_state::AbstractVector{Float64}; radius::Int=1)
    N = length(λ_state)
    λ_smooth = similar(λ_state)

    for i in 1:N
        vals = Float64[]
        for r in -radius:radius
            j = mod1(i + r, N)
            push!(vals, λ_state[j])
        end
        λ_smooth[i] = mean(vals)
    end

    return λ_smooth
end

function hybrid_mean_statewise(x_lor::AbstractVector, x_ml::AbstractVector, λ_state::AbstractVector)
    return (1.0 .- λ_state) .* x_lor .+ λ_state .* x_ml
end




end