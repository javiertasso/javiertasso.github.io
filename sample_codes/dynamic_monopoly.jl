using LinearAlgebra 
using Optim
using Plots 
using Interpolations
using Distributions
using DifferentialEquations

# ----------------------------------------------------------- #
## Dynamic Monopoly ----------------------------------------- #
# ----------------------------------------------------------- #

# Parameters
c  = 0.10
α  = 0.75
β  = 0.96
γ  = 0.25

# Functions
function period_profits(Q, d) 
    p = d / Q^α
    return p * Q - c * Q 
end

function tauchen(N, μ, σ, ρ, k = 3)
    
    # Grid for ln(D)
    lnD_min = μ - k * σ / sqrt(1 - ρ^2)
    lnD_max = μ + k * σ / sqrt(1 - ρ^2)
    lnD = range(lnD_min, lnD_max, length = N)
    step = lnD[2] - lnD[1]

    # Transition matrix
    Pi = zeros(N, N)
    for i in 1:N
        for j in 1:N
            if j == 1
                Pi[i, j] =     cdf(Normal(lnD[i] * ρ, σ), lnD[j] + step / 2)
            elseif j == N
                Pi[i, j] = 1 - cdf(Normal(lnD[i] * ρ, σ), lnD[j] - step / 2)
            else
                Pi[i, j] =     cdf(Normal(lnD[i] * ρ, σ), lnD[j] + step / 2) -
                               cdf(Normal(lnD[i] * ρ, σ), lnD[j] - step / 2)
            end
        end
    end

    return exp.(lnD), Pi
end

N      = 7
D_vals, Π = tauchen(N, 0.0, 0.10, 0.75, 3)
Q_vals = ((1-α) .* D_vals ./c).^(1/α)

# Create a sensible grid 
min_q = minimum(Q_vals)
max_q = maximum(Q_vals)
q_grid = range(min_q, max_q, length = 101)
V_guess = [[0.0 for q in q_grid, d in D_vals], collect(q_grid), D_vals]

function vfi(tol_level, max_it, β, γ, V_guess, Π)

    N      = length(V_guess[3])
    D      = V_guess[3]
    Q      = V_guess[2]
    diff   = 1.0 
    V_prev = deepcopy(V_guess)
    V_next = deepcopy(V_guess)
    p_func = deepcopy(V_guess)

    for it in 1:max_it 

        V_interp = [linear_interpolation((V_prev[2]), V_prev[1][:,n]) for n in 1:N]
        
        for ((j, d), (k, q)) in Iterators.product(enumerate(D), enumerate(Q))

            Π_row = Π[j, :]

            function obj(q′) 
                period = period_profits(q′, d) - γ/2 * (q′ - q)^2
                cont   = dot(Π_row, [V_interp[n](q′) for n in 1:N])
                return -(period + β * cont)
            end

            res = optimize(obj, min_q, max_q)
            V_next[1][k, j] = - Optim.minimum(res)
            p_func[1][k, j] = Optim.minimizer(res)

        end

        diff = maximum(abs.(V_next[1] - V_prev[1]))
        V_prev = deepcopy(V_next)

        if diff < tol_level
            return V_next, p_func, it, diff
        end

        if it == max_it
            println("No convergence")
            return V_next, p_func, it, diff
        end
        
    end 

end 

value, policy, it, diff  = vfi(10^-9, 1000, β, γ, V_guess, Π)

plot(value[2], (1-β) .* value[1][:,4], label = "D = $(round(D_vals[4], digits = 2))", lw = 2)
plot( policy[2], policy[1][:,4], label = "D = $(round(D_vals[4], digits = 2))", lw = 2, title = "Policy Function", xlabel = "Q", ylabel = "Q'(Q, D)")
plot!(policy[2], policy[1][:,1], label = "D = $(round(D_vals[1], digits = 2))", lw = 2)
plot!(policy[2], policy[1][:,7], label = "D = $(round(D_vals[7], digits = 2))", lw = 2)
plot!(policy[2], policy[2], label = nothing, lw = 1, linestyle = :dash, color = :black)  

# ----------------------------------------------------------- #
# ----------------------------------------------------------- #


## Solow Model 

# Parameters
α = 0.3           # Output elasticity of capital
s = 0.2           # Savings rate
δ = 0.05          # Depreciation rate
n = 0.01          # Population growth rate
k0 = 1.0          # Initial capital per worker
T = 100.0         # Time horizon

# Production function
f(k) = k^α

# Differential equation: dk/dt = s*f(k) - (δ + n)*k
function solow_model!(du, u, p, t)
    k = u[1]  # Capital per worker
    du[1] = s * f(k) - (δ + n) * k
end

# Initial condition and time span
u0 = [k0]
tspan = (0.0, T)

# Solve the ODE
prob = ODEProblem(solow_model!, u0, tspan)
sol = solve(prob, Tsit5())  # Tsit5 is a common ODE solver

# Extract the results
time = sol.t
capital = sol.u

k_star = (s / (δ + n))^(1 / (1 - α)) # Steady-state capital
k_analytical = [k_star + (k0 - k_star) * exp(- (δ + n) * (1 - α) * t) for t in time]

plot(time, [k[1] for k in capital], label="Capital per worker (k)", xlabel="Time", ylabel="Capital", title="Solow Model Dynamics")
plot!(time, k_analytical, label="Analytical solution", linestyle=:dash, color=:black)

# ----------------------------------------------------------- #
# Ramsey-Cass-Koopmans model 
# ----------------------------------------------------------- #


# Parameters
α = 0.33       # Output elasticity of capital
θ = 2.0        # Relative risk aversion
δ = 0.05       # Depreciation rate
ρ = 0.03       # Discount rate
n = 0.02       # Population growth rate
s = 0.2        # Savings rate (initial guess)

k_star = (α / (δ + n + ρ * θ))^(1 / (1 - α))
C_star = k_star^α - (δ + n) * k_star

# Initial conditions
k0 = 0.8 * k_star       # Initial capital per worker
C0 = 0.8 * C_star     # Initial consumption per worker
T = 100.0      # Time horizon



# Differential equations
function ramsey_system!(du, u, p, t)
    k, C = u
    dk = k^α - C - (δ + n) * k
    dC = (α * k^(α-1) - δ - ρ) * C / θ
    du[1] = dk
    du[2] = dC
end

# Solve the system
u0 = [k0, C0]
tspan = (0.0, T)
prob = ODEProblem(ramsey_system!, u0, tspan)
sol = solve(prob, Tsit5())  # Use the Tsitouras 5/4 method (adaptive Runge-Kutta)

# Plot the results
plot(sol, xlabel="Time", ylabel="Values", label=["Capital per worker (k)" "Consumption per worker (C)"], title="Ramsey-Cass-Koopmans Model Dynamics")

plot(sol.t, [k[1] for k in sol.u], label="Capital per worker (k)", xlabel="Time", ylabel="Values", title="Ramsey-Cass-Koopmans Model Dynamics")
plot(sol.t, [c[2] for c in sol.u], label="Consumption per worker (C)", linestyle=:dash, color=:black)