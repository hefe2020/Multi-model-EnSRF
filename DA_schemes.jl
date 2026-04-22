module DA_schemes
export stochastic_enkf, global_ensrf, serial_ensrf

using Statistics
using LinearAlgebra
using Random
using Distributions


function stochastic_enkf(E::AbstractMatrix{Float64},
                         H::AbstractMatrix{Float64},
                         R::AbstractMatrix{Float64},
                         y::AbstractMatrix{Float64};
                         rho::Union{AbstractMatrix{Float64}, Nothing}=nothing,
                         inflation::Float64=1.0,
                         rng::AbstractRNG=Random.default_rng())

    _, Ne = size(E)

    xbar = mean(E, dims=2)
    A = E .- xbar

    if rho === nothing
        P = inflation .* (A * A') ./ (Ne - 1)
    else
        P = inflation .* (rho .* (A * A')) ./ (Ne - 1)
    end

    S = H * P * H' + R
    K = P * H' * inv(S)

    obs_dist = MvNormal(zeros(size(R, 1)), Symmetric(R))
    E_a = similar(E)

    for i in 1:Ne
        y_pert = y[:, 1] .+ rand(rng, obs_dist)
        E_a[:, i] = E[:, i] .+ K * (y_pert .- H * E[:, i])
    end

    return E_a
end

function global_ensrf(E::AbstractMatrix{Float64},
                      H::AbstractMatrix{Float64},
                      R::AbstractMatrix{Float64},
                      y::AbstractMatrix{Float64},
                      rho::Union{AbstractMatrix{Float64}, Nothing}=nothing,
                      inflation::Float64=1.0)

    _, Ne = size(E)

    xbar = mean(E, dims=2)
    A = E .- xbar

    if rho === nothing
        P = inflation .* (A * A') ./ (Ne - 1)
    else
        P = inflation .* (rho .* (A * A')) ./ (Ne - 1)
    end

    S = H * P * H' + R
    K = P * H' * inv(S)

    xbar_a = xbar .+ K * (y .- H * xbar)
    T = real((I + P * H' * inv(R) * H)^(-1/2))
    A_a = T * A

    return xbar_a .+ A_a
end

function serial_ensrf(E_f::AbstractMatrix{Float64},
                      H::AbstractMatrix{Float64},
                      R::AbstractMatrix{Float64},
                      y::AbstractMatrix{Float64},
                      rho::AbstractMatrix{Float64})

    N, Ne = size(E_f)
    m = size(H, 1)

    xbar = vec(mean(E_f, dims=2))
    A = E_f .- reshape(xbar, :, 1)

    for j in 1:m
        hj = vec(H[j, :])
        rj = R[j, j]

        Pf = (A * A') / (Ne - 1)
        Pf_loc = rho .* Pf
        Pf_loc = 0.5 .* (Pf_loc .+ Pf_loc')

        ybar_j = dot(hj, xbar)
        Yj = vec(hj' * A)
        dj = y[j, 1] - ybar_j

        s2 = dot(hj, Pf_loc * hj) + rj
        if !isfinite(s2) || s2 <= 1e-12
            continue
        end

        K = (Pf_loc * hj) ./ s2
        alpha = 1.0 / (1.0 + sqrt(rj / s2))

        xbar = xbar .+ K .* dj
        A = A .- alpha .* (K * transpose(Yj))

        if !(all(isfinite, xbar) && all(isfinite, A))
            error("Non-finite values inside EnSRF at obs index $j")
        end
    end

    return reshape(xbar, :, 1) .+ A
end

end