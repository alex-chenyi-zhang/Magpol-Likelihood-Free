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


function compute_summary_stats!(ss::Matrix{Float64}, sp_conf::Matrix{Int}, feats::Matrix{Float64})
    n_s = size(ss,2) #number of samples 
    n_m = size(sp_conf,1)
    for i_sample in 1:n_s
        #ss[1,i_sample] = mean(sp_conf[:,i_sample])
        #ss[2,i_sample] = mean(sp_conf[:,i_sample].*feats)
        ss[1,i_sample]=0
        ss[2,i_sample]=0
        for i_m in 1:n_m
            ss[1,i_sample] += sp_conf[i_m,i_sample]
            ss[2,i_sample] += sp_conf[i_m,i_sample]*feats[i_m,1]
            ss[3,i_sample] += sp_conf[i_m,i_sample]*feats[i_m,2]
        end
        ss[1,i_sample] = ss[1,i_sample]/n_m
        ss[2,i_sample] = ss[2,i_sample]/n_m
        ss[3,i_sample] = ss[3,i_sample]/n_m
        ss[4,i_sample] = lag_p_autocovariance(sp_conf[:,i_sample], 3)
    end
end


function log_synth_likelihood(mat::Matrix{Float64}, avg::Array{Float64}, data::Array{Float64})
    s = -0.5*(avg .- data)'*inv(mat)*(avg .- data) -0.5*log(abs(det(mat)))
    return s
end

#####################################################################################
#####################################################################################


## for now this is just the backbone of the function to carry out the likelihood-free inference task
function synthetic_likelihood_polymer(n_samples::Int, sample_lag::Int, n_params::Int, initial_weights::Array{Float64}, features_file::String)
    io = open(features_file, "r")
    features = readdlm(io,Float64)
    close(io)
    n_mono = size(features, 1)
    n_feats = size(features, 2)

    #data = [0.6036745406824147, -0.3997000374953125, -0.3992614680299055, 0.06693990928046743] # 2, -1.5

    io = open("data_file.txt", "r")
    data = readdlm(io,Float64)
    close(io)
    println(data,"\n\n")
    n_data = size(data, 1)
    n_ss = size(data, 2) # number of summary statistics
    println(n_ss, "\n\n")

    accepted_moves = 0
    stride = 100
    n_strides = cld(n_samples*sample_lag, stride) # integer ceiling of the division
    spins_coupling = 1.0
    alpha = 0.5
    inv_temps = [1.0, 0.9, 0.8, 0.74, 0.65, 0.62, 0.6, 0.55, 0.5, 0.4]
    n_temps = length(inv_temps)

    spins_configs = zeros(Int, n_mono, n_samples)
    
    
    
    polymers = Array{mp.Magnetic_polymer}(undef, n_temps)
    trajs = Array{mp.MC_data}(undef, n_temps)
    for i_temp in 1:n_temps
        polymers[i_temp] = mp.Magnetic_polymer(n_mono, inv_temps[i_temp], spins_coupling, alpha)
        trajs[i_temp] = mp.MC_data(n_strides*stride)
        if isfile("simulation_data/final_config_$(n_mono)_$(inv_temps[1]).txt")
            mp.initialize_poly!(polymers[i_temp],"simulation_data/final_config_$(n_mono)_$(inv_temps[1]).txt")
        else
            mp.initialize_poly!(polymers[i_temp])
        end
    end 

    cov_mat = zeros(n_ss,n_ss)
    #inv_cov_mat = zeros(3,3)
    delta_w = 0.1
    param_series = zeros(2,n_params)
    SL_series = zeros(n_params)
    ss_mean_series = zeros(n_ss, n_params)
    #ss_cov_determinant_series = zeros(n_params)
    summary_stats = zeros(n_ss, n_samples)
    weights = zeros(2)
    trial_weights = zeros(2)
    weights .= initial_weights

    fields = zeros(n_mono)
    for i in 1:n_feats
        fields .+= features[:,i] .* weights[i]
    end

    mp.set_fields!(polymers, fields)
    mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps,spins_configs,sample_lag,n_samples)
    compute_summary_stats!(summary_stats, spins_configs, features)
    ss_mean_series[:,1] .= vec(mean(summary_stats, dims=2))
    cov_mat .= cov(summary_stats, dims=2)

    syn_like = 0.0
    for i_data in 1:n_data
        syn_like += log_synth_likelihood(cov_mat, ss_mean_series[:,1], data[i_data,:])
    end
    
    param_series[1,1] = weights[1]
    param_series[2,1] = weights[2]
    SL_series[1] = syn_like 
    w_acceptance = 0.0
    
    for i_param in 2:n_params
        #=trial_weights .= weights
        if rand((1,2))==1
            trial_weights[1] += (2*rand()-1)*delta_w
        else
            trial_weights[2] += (2*rand()-1)*delta_w
        end=#
            
        trial_weights .= weights .+ (2 .* rand(2) .-1) .*delta_w
        
        
        fields = zeros(n_mono)
        for i in 1:n_feats
            fields .+= features[:,i] .* trial_weights[i]
        end
        mp.set_fields!(polymers, fields)
        mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps,spins_configs,sample_lag,n_samples)
        compute_summary_stats!(summary_stats, spins_configs, features)

        ss_mean_series[:,i_param] .= vec(mean(summary_stats, dims=2))
        cov_mat .= cov(summary_stats, dims=2)

        trial_syn_like = 0.0
        for i_data in 1:n_data
            trial_syn_like += log_synth_likelihood(cov_mat, ss_mean_series[:,i_param], data[i_data,:])
        end
        
        delta_syn_like = trial_syn_like -syn_like
        delta_syn_like>=0 ? w_acceptance=1 : w_acceptance=exp(delta_syn_like)
        if i_param%10 == 0
            println(i_param)
            println("w1: ",weights[1]," ---> ",trial_weights[1])
            println("w2: ",weights[2]," ---> ",trial_weights[2])
            println("delta_syn_like: ", delta_syn_like)
            println("acceptance: ", w_acceptance)
        end
        if w_acceptance > rand()
            weights .= trial_weights
            syn_like = trial_syn_like
            accepted_moves += 1
        end
        param_series[1,i_param] = weights[1]
        param_series[2,i_param] = weights[2]
        SL_series[i_param] = syn_like
    end

    
    !isdir("SL_data") && mkdir("SL_data")
    open("SL_data/weights.txt", "w") do io
        writedlm(io, param_series)
    end
    open("SL_data/syn_likes.txt", "w") do io
        writedlm(io, SL_series)
    end
    mp.write_results(polymers[1],trajs[1])

end

#####################################################################################
#####################################################################################

function generate_data(features_file::String, n_strides::Int, weights::Array{Float64})
    io = open(features_file, "r")
    features = readdlm(io,Float64)
    close(io)
    n_mono = size(features, 1)
    n_feats = size(features, 2)

    spins_coupling = 1.0
    alpha = 0.5
    inv_temps = [1.0, 0.9, 0.8, 0.74, 0.65, 0.62, 0.6, 0.55, 0.5, 0.4]
    n_temps = length(inv_temps)
    stride = 500

    polymers = Array{mp.Magnetic_polymer}(undef, n_temps)
    trajs = Array{mp.MC_data}(undef, n_temps)
    for i_temp in 1:n_temps
        polymers[i_temp] = mp.Magnetic_polymer(n_mono, inv_temps[i_temp], spins_coupling, alpha)
        trajs[i_temp] = mp.MC_data(n_strides*stride)
        if isfile("simulation_data/final_config_$(n_mono)_$(inv_temps[1]).txt")
            mp.initialize_poly!(polymers[i_temp],"simulation_data/final_config_$(n_mono)_$(inv_temps[1]).txt")
        else
            mp.initialize_poly!(polymers[i_temp])
        end
    end 

    fields = zeros(n_mono)
    for i in 1:n_feats
        fields .+= features[:,i] .* weights[i]
    end
    mp.set_fields!(polymers, fields)

    mp.MMC_run!(polymers, trajs, n_strides, stride, inv_temps)
    mp.MMC_run!(polymers, trajs, n_strides, stride, inv_temps)
    mp.MMC_run!(polymers, trajs, n_strides, stride, inv_temps)
    
    n_data = 10
    summary_stats = zeros(4,n_data)
    spins_conf = zeros(Int,n_mono,n_data)
    for i_data in 1:n_data
        mp.MMC_run!(polymers, trajs, n_strides, stride, inv_temps)
        #mp.write_results(polymers[1],trajs[1])
        spins_conf[:,i_data] .= polymers[1].spins
    end
    compute_summary_stats!(summary_stats,spins_conf,features)
    println(summary_stats)
end

