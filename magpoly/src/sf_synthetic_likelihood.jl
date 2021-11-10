include("magpoly.jl")

using .magpoly
const mp = magpoly
using DelimitedFiles, Random, Distributions, LinearAlgebra
#= This file contains the structures and functions to do carry
   out the inference using synthetic likelihood =#

struct summary_stats{T1<:Int, T2<:Float64}
    n_samples::T1
    magnetizations::T2
    spin_feature_overlaps::T2
    lag_p_autocorrelation::T2
end

function summary_stats(n_samples::Int)
    summary_stats(n_samples,zeros(n_samples),zeros(n_samples),zeros(n_samples))
end

#####################################################################################
# This is the structure for the parameter samples generated through synthetic likelihood
struct SL_data{T1<:Int, T2<:Float64}  
    n_samples::T1
    n_params::T1
    param_values::T2
end

function SL_data(n_samples::Int, n_params::Int)
    SL_data(n_samples, n_params, zeros(n_params, n_samples))
end

#####################################################################################
#####################################################################################
function lag_p_autocovariance(vect::Array{Int}, p::Int)
    avg = 0.0
    avg_lag = 0.0
    var = 0.0
    var_lag = 0.0
    corr = 0.0
    n = length(vect)
    for i in 1:(n-p)
        avg += vect[i]
        avg_lag += vect[i+p]
        corr += vect[i]*vect[i+p]
    end
    avg = avg/(n-p)
    avg_lag = avg_lag/(n-p)
    corr = corr/(n-p)
    for i in 1:(n-p)
        var += (vect[i]-avg)^2
        var_lag += (vect[i+p]-avg_lag)^2
    end
    var = var/(n-p-1)
    var_lag = var_lag/(n-p-1)
    return (corr - avg*avg_lag)/(sqrt(var*var_lag))
end


function compute_summary_stats!(ss::Matrix{Float64}, sp_conf::Matrix{Int}, feats::Array{Float64})
    n_s = size(ss,2) #number of samples 
    n_m = size(sp_conf,1)
    for i_sample in 1:n_s
        #ss[1,i_sample] = mean(sp_conf[:,i_sample])
        #ss[2,i_sample] = mean(sp_conf[:,i_sample].*feats)
        ss[1,i_sample]=0
        ss[2,i_sample]=0
        for i_m in 1:n_m
            ss[1,i_sample] += sp_conf[i_m,i_sample]
            ss[2,i_sample] += sp_conf[i_m,i_sample]*feats[i_m]
        end
        ss[1,i_sample] = ss[1,i_sample]/n_m
        ss[2,i_sample] = ss[2,i_sample]/n_m
        ss[3,i_sample] = lag_p_autocovariance(sp_conf[:,i_sample], 3)
    end
end


function log_synth_likelihood(mat::Matrix{Float64}, avg::Array{Float64}, data::Array{Float64})
    s = -0.5*(avg .- data)'*inv(mat)*(avg .- data) -0.5*log(abs(det(mat)))
    #println(log(abs(det(mat))))
    #println((avg .- data)'*inv(mat)*(avg .- data))
    return s
end

#####################################################################################
#####################################################################################


## for now this is just the backbone of the function to carry out the likelihood-free inference task
function synthetic_likelihood_polymer(n_samples::Int, n_params::Int, sample_lag::Int, initial_weight::Float64, features_file::String)
    io = open(features_file, "r")
    features = readdlm(io,Float64)
    close(io)
    n_mono = size(features, 1)

    data = [0.662818, -0.0552691, -0.0162029]
    accepted_moves = 0
    stride = 50
    n_strides = cld(n_samples*sample_lag, stride) # integer ceiling of the division
    spins_coupling = 1.0
    alpha = 0.5
    inv_temps = [1.0, 0.9, 0.8, 0.74, 0.65, 0.62, 0.6, 0.55, 0.5, 0.4]
    n_temps = length(inv_temps)

    spins_configs = zeros(Int, n_mono, n_samples)
    cov_mat = zeros(3,3)
    inv_cov_mat = zeros(3,3)
    
    
    polymers = Array{mp.Magnetic_polymer}(undef, n_temps)
    trajs = Array{mp.MC_data}(undef, n_temps)
    for i_temp in 1:n_temps
        polymers[i_temp] = mp.Magnetic_polymer(n_mono, inv_temps[i_temp], spins_coupling, alpha)
        trajs[i_temp] = mp.MC_data(n_strides*stride)
        mp.initialize_poly!(polymers[i_temp])#,"simulation_data/final_config_$(n_mono)_$(inv_temps[i_temp]).txt")
    end 

    n_ss = 3 #number of summary statistics
    delta_w = 0.3
    param_series = zeros(n_params)
    SL_series = zeros(n_params)
    ss_mean_series = zeros(n_ss, n_params)
    #ss_cov_determinant_series = zeros(n_params)
    summary_stats = zeros(n_ss, n_samples)
    weight = initial_weight

    mp.set_fields!(polymers, weight .* features)
    mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps,spins_configs,sample_lag,n_samples)
    compute_summary_stats!(summary_stats, spins_configs, features)
    ss_mean_series[:,1] .= vec(mean(summary_stats, dims=2))
    cov_mat .= cov(summary_stats, dims=2)
    #determ = det(cov_mat)
    #inv_cov_mat = inv(cov_mat)
    syn_like = log_synth_likelihood(cov_mat, ss_mean_series[:,1], data)
    param_series[1] = weight
    SL_series[1] = syn_like 
    w_acceptance = 0.0
    #println(size(vec(mean(summary_stats, dims=2))))
    #println(size(ss_mean_series[:,1]))
    #println(size(ss_mean_series[:,1] .- data))
    #println(size((ss_mean_series[:,1] .- data)'*inv(cov_mat)*(ss_mean_series[:,1] .- data)))
    
    for i_param in 2:n_params
        trial_weight = weight + (2*rand()-1)*delta_w
        mp.set_fields!(polymers, trial_weight .* features)
        mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps,spins_configs,sample_lag,n_samples)
        compute_summary_stats!(summary_stats, spins_configs, features)

        #println(mean(summary_stats, dims=2))
        ss_mean_series[:,i_param] .= vec(mean(summary_stats, dims=2))
        cov_mat .= cov(summary_stats, dims=2)
        trial_syn_like = log_synth_likelihood(cov_mat, ss_mean_series[:,i_param], data)
        delta_syn_like = trial_syn_like -syn_like
        delta_syn_like>=0 ? w_acceptance=1 : w_acceptance=exp(delta_syn_like)
        if i_param%100 == 0
            println(i_param, "\n weight = ", weight, "\n accept = ",w_acceptance, "\n syn_like = ", syn_like)
            println("trial_weight: ",trial_weight)
            println("delta_syn_like: ", delta_syn_like)
        end
        if w_acceptance > rand()
            weight = trial_weight
            syn_like = trial_syn_like
            accepted_moves += 1
        end
        param_series[i_param] = weight
        SL_series[i_param] = syn_like
    end

    
    !isdir("SL_data") && mkdir("SL_data")
    open("SL_data/weights.txt", "w") do io
        writedlm(io, param_series)
    end
    open("SL_data/syn_likes.txt", "w") do io
        writedlm(io, SL_series)
    end

end
