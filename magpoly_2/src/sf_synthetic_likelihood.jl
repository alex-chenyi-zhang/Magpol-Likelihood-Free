include("magpoly.jl")

using .magpoly
const mp = magpoly
using DelimitedFiles, Random, Distributions, LinearAlgebra
#= This file contains the structures and functions to do carry
   out the inference using synthetic likelihood =#

#=struct summary_stats{T1<:Int, T2<:Float64}
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
end=#


function compute_summary_stats!(ss::Matrix{Float64}, sp_conf::Matrix{Int}, feats::Matrix{Float64})
    n_s = size(ss,2) #number of samples 
    n_m = size(sp_conf,1)
    n_feats = size(feats, 2)
    for i_sample in 1:n_s
        for i_feat in 1:n_feats
            ss[i_feat,i_sample]=0
        end
        for i_m in 1:n_m
            for i_feat in 1:n_feats
                ss[i_feat,i_sample] += sp_conf[i_m,i_sample]*feats[i_m, i_feat]
            end
        end
        for i_feat in 1:n_feats
            ss[i_feat,i_sample] = ss[i_feat,i_sample]/n_m
        end
    end
end


function log_synth_likelihood(mat::Matrix{Float64}, avg::Array{Float64}, data::Array{Float64})
    s = -0.5*(avg .- data)'*inv(mat)*(avg .- data) -0.5*log(abs(det(mat)))
    return s
end

#####################################################################################
#####################################################################################


## for now this is just the backbone of the function to carry out the likelihood-free inference task
function synthetic_likelihood_polymer(n_samples::Int, sample_lag::Int, stride::Int, n_params::Int, delta_w::Float64,initial_weights::Array{Float64}, features_file::String, data_file::String)
    io = open(features_file, "r")
    features = readdlm(io,Float64)
    close(io)
    n_mono = size(features, 1)
    n_feats = size(features, 2)

    io = open(data_file, "r")
    data = readdlm(io,Float64; header=true)[1][1:end,:]
    close(io)
    println(data,"\n\n")
    n_data = size(data, 1)
    n_ss = size(data, 2) # number of summary statistics
    println(n_ss, "\n\n")

    accepted_moves = 0
    n_strides = cld(n_samples*sample_lag, stride) # integer ceiling of the division
    spins_coupling = 0.5
    inv_temps = [1.0, 0.9, 0.8, 0.74, 0.65, 0.62, 0.6, 0.55, 0.5, 0.4]
    n_temps = length(inv_temps)

    spins_configs = zeros(Int, n_mono, n_samples)
    
    
    
    polymers = Array{mp.Magnetic_polymer}(undef, n_temps)
    trajs = Array{mp.MC_data}(undef, n_temps)
    for i_temp in 1:n_temps
        polymers[i_temp] = mp.Magnetic_polymer(n_mono)#, inv_temps[i_temp], spins_coupling, alpha)
        trajs[i_temp] = mp.MC_data(n_strides*stride)
        if isfile("simulation_data/final_config_$(n_mono).txt")
            mp.initialize_poly!(polymers[i_temp],"simulation_data/final_config_$(n_mono).txt")
        else
            mp.initialize_poly!(polymers[i_temp])
        end
    end 

    cov_mat = zeros(n_ss,n_ss)
    #inv_cov_mat = zeros(3,3)
    #delta_w = 0.05
    param_series = zeros(n_feats,n_params)
    SL_series = zeros(n_params)
    ss_mean_series = zeros(n_ss, n_params)
    #ss_cov_determinant_series = zeros(n_params)
    summary_stats = zeros(n_ss, n_samples)
    weights = zeros(n_feats)
    trial_weights = zeros(n_feats)
    weights .= initial_weights

    fields = zeros(n_mono)
    for i in 1:n_feats
        fields .+= features[:,i] .* weights[i]
    end

    #mp.set_fields!(polymers, fields)
    for i_eq in 1:20
        mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps, spins_coupling, fields)
    end

    mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps,spins_configs,sample_lag,n_samples, spins_coupling, fields)
    compute_summary_stats!(summary_stats, spins_configs, features)
    ss_mean_series[:,1] .= vec(mean(summary_stats, dims=2))
    cov_mat .= cov(summary_stats, dims=2)

    syn_like = 0.0
    for i_data in 1:n_data
        syn_like += log_synth_likelihood(cov_mat, ss_mean_series[:,1], data[i_data,:])
    end
    
    param_series[1,1] = weights[1]
    param_series[2,1] = weights[2]
    param_series[3,1] = weights[3]
    SL_series[1] = syn_like 
    w_acceptance = 0.0
    
    for i_param in 2:n_params
        #=trial_weights .= weights
        if rand((1,2))==1
            trial_weights[1] += (2*rand()-1)*delta_w
        else
            trial_weights[2] += (2*rand()-1)*delta_w
        end=#
            
        trial_weights .= weights .+ (2 .* rand(n_feats) .-1) .*delta_w
        
        
        fields = zeros(n_mono)
        for i in 1:n_feats
            fields .+= features[:,i] .* trial_weights[i]
        end
        #mp.set_fields!(polymers, fields)

        mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps, spins_coupling, fields) # This is an short equilibration run so that the expectations are a bit better when changing weights
        mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps,spins_configs,sample_lag,n_samples, spins_coupling, fields)
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
            println("w3: ",weights[3]," ---> ",trial_weights[3])
            println("delta_syn_like: ", delta_syn_like)
            println("acceptance: ", w_acceptance)
            println("avg acceptance: ", accepted_moves/i_param)
        end
        if w_acceptance > rand()
            weights .= trial_weights
            syn_like = trial_syn_like
            accepted_moves += 1
        end
        param_series[1,i_param] = weights[1]
        param_series[2,i_param] = weights[2]
        param_series[3,i_param] = weights[3]
        SL_series[i_param] = syn_like
    end
    println("Acceptance ratio: ", accepted_moves/n_params)

    
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


function amhi_polymer(n_samples::Int, sample_lag::Int, stride::Int, n_params::Int, delta_theta::Float64,initial_theta::Array{Float64}, features_file::String, data_file::String)
    # Read features from file
    io = open(features_file, "r")
    features = readdlm(io,Float64)
    close(io)
    n_mono = size(features, 1)
    n_feats = size(features, 2)

    # Read generated data. In this algo they're not summary stats but the full data!
    io = open(data_file,"r")
    data_spins = readdlm(io, Int64; header=true)[1][1:end,:]
    close(io)
    n_data = size(data_spins, 1)
    println(n_mono)
    println(n_data)
    println(typeof(data_spins))


    accepted_moves = 0
    n_strides = cld(n_samples*sample_lag, stride) # integer ceiling of the division
    #spins_coupling = 0.5

    inv_temps = [1.0, 0.9, 0.8, 0.74, 0.65, 0.62, 0.6, 0.55, 0.5, 0.4]
    n_temps = length(inv_temps)

    spins_configs = zeros(Int, n_mono, n_samples) # where you store the generated data
    ising_energies = zeros(n_samples)
    
    ########################################################### Initialize some quantitites in this section
    polymers = Array{mp.Magnetic_polymer}(undef, n_temps)
    trajs = Array{mp.MC_data}(undef, n_temps)
    for i_temp in 1:n_temps
        polymers[i_temp] = mp.Magnetic_polymer(n_mono)
        trajs[i_temp] = mp.MC_data(n_strides*stride)
        if isfile("simulation_data/final_config_$(n_mono).txt")
            mp.initialize_poly!(polymers[i_temp],"simulation_data/final_config_$(n_mono).txt")
        else
            mp.initialize_poly!(polymers[i_temp])
        end
    end 

    
    param_series = zeros(n_feats+1,n_params)
    overlaps = zeros(n_feats+1, n_samples)
    avg_overlaps = zeros(n_feats+1)
    #cov_overlaps = zeros(n_feats+1, n_feats+1)
    
    theta = zeros(n_feats+1)    #from theta[1] to theta[n_feats] we have the regression weights, theta[n_feats+1] is the spins coupling
    trial_theta = zeros(n_feats+1)
    theta .= initial_theta
    if theta[n_feats+1] <= 0 || theta[n_feats+1] > 1
        theta[n_feats+1] = 0.5
    end
    ###
    #theta[n_feats+1] = 0.5
    ###

    fields = zeros(n_mono)
    for i in 1:n_feats
        fields .+= features[:,i] .* theta[i]
    end
    #############################################################
    
    
    for i_eq in 1:20
        mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps, theta[n_feats+1], fields) # theta[n_feats+1 is the coupling]
    end
    
    param_series[:,1] .= theta
    acceptance = 0.0

    energy = 0
    for i_data in 1:n_data
        for i_mono in 1:n_mono
            energy -= fields[i_mono]*data_spins[i_data,i_mono]
        end
    end

    
    energy_correction = 0.0

    ##
    theta_variation = zeros(n_feats+1)
    ##

    for i_param in 2:n_params
        
        theta_variation .= (2 .* rand(n_feats+1) .-1) .*delta_theta
        ###
        #theta_variation[n_feats+1] = 0.0
        ###
        trial_theta .= theta .+ theta_variation
        
    
        if trial_theta[n_feats+1]>0 && trial_theta[n_feats+1]<= 1.0
            trial_fields = zeros(n_mono)
            for i in 1:n_feats
                trial_fields .+= features[:,i] .* trial_theta[i]
            end
            
            trial_energy = 0
            for i_data in 1:n_data
                for i_mono in 1:n_mono
                    trial_energy -= trial_fields[i_mono]*data_spins[i_data,i_mono]
                end
            end

            ##################################################

            mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps, theta[n_feats+1], fields) # This is an short equilibration run so that the expectations are a bit better when changing weights
            mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps,spins_configs,ising_energies,sample_lag,n_samples, theta[n_feats+1], fields)
            
            
            fill!(overlaps, 0.0)
            for i_sample in 1:n_samples
                for i_feat in 1:n_feats
                    for i_mono in 1:n_mono
                        overlaps[i_feat, i_sample] += spins_configs[i_mono, i_sample]*features[i_mono, i_feat]
                    end
                end
                overlaps[n_feats+1, i_sample] = ising_energies[i_sample]
            end

            
            avg_overlaps = vec(mean(overlaps, dims=2))
            
            energy_correction = 0
            #linear correction
            for j in 1:n_feats+1
                energy_correction += theta_variation[j]*avg_overlaps[j]
            end
            energy_correction = energy_correction * n_data
            ##################################################
            
            energy_correction_J = 0.0
            for i_data in 1:n_data
                mp.set_spins!(polymers, data_spins[i_data,:])
                # here I run the MC's with the extra 'true' argument to quench the spins
                mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps, theta[n_feats+1], fields, true) # This is an short equilibration run so that the expectations are a bit better when changing weights
                mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps,spins_configs,ising_energies,sample_lag,n_samples, theta[n_feats+1], fields, true)
                energy_correction_J += mean(ising_energies) * theta_variation[n_feats+1]
            end

            
            ##################################################
            delta_acc = -(trial_energy - energy) - energy_correction + energy_correction_J
            delta_acc>=0 ? acceptance=1 : acceptance=exp(delta_acc)
            if i_param%10 == 0
                println(i_param)
                println("w1: ",theta[1]," ---> ",trial_theta[1])
                println("w2: ",theta[2]," ---> ",trial_theta[2])
                #println("w3: ",theta[3]," ---> ",trial_theta[3])
                println("J:  ",theta[3]," ---> ",trial_theta[3])
                println("delta_energy: ", -(trial_energy-energy))
                println("delta_acc: ", delta_acc)
                println("order of error: ", - sum(theta_variation.^2) *n_data)
                println("acceptance: ", acceptance)
                println("avg acceptance: ", accepted_moves/i_param)
            end
            if acceptance > rand()
                theta .= trial_theta
                energy = trial_energy
                #resample_needed = true
                fields .= trial_fields
                accepted_moves += 1
            end
        end

        param_series[1,i_param] = theta[1]
        param_series[2,i_param] = theta[2]
        param_series[3,i_param] = theta[3]
        #param_series[4,i_param] = theta[4]
    end
    println("Acceptance ratio: ", accepted_moves/n_params)

 
    !isdir("AMHI_data") && mkdir("AMHI_data")
    open("AMHI_data/thetas_$(n_data)amh$(n_samples)_$(delta_theta).txt", "w") do io
        writedlm(io, param_series)
    end
    
    mp.write_results(polymers[1],trajs[1])

end

#####################################################################################

# QAMHI: Approximate metropolis hastings inference with Quadratic corrections
function Qamhi_polymer(n_samples::Int, sample_lag::Int, stride::Int, n_params::Int, delta_theta::Float64,initial_theta::Array{Float64}, features_file::String, data_file::String)
    # Read features from file
    println("gaussian")
    io = open(features_file, "r")
    features = readdlm(io,Float64)
    close(io)
    n_mono = size(features, 1)
    n_feats = size(features, 2)
    println("Number of features: " ,n_feats)

    # Read generated data. In this algo they're not summary stats but the full data!
    io = open(data_file,"r")
    data_spins = readdlm(io, Int64; header=true)[1][1:end,:]
    close(io)
    n_data = size(data_spins, 1)
    println(n_mono)
    println(n_data)
    println(typeof(data_spins))


    accepted_moves = 0
    n_strides = cld(n_samples*sample_lag, stride) # integer ceiling of the division
    #spins_coupling = 0.5
    
    #inv_temps = [1.0, 0.9, 0.8, 0.74, 0.65, 0.62, 0.6, 0.55, 0.5, 0.4]
    inv_temps = [1.0, 0.96, 0.92, 0.89, 0.85, 0.83, 0.82, 0.79, 0.75, 0.67, 0.57, 0.5, 0.4]
    n_temps = length(inv_temps)

    spins_configs = zeros(Int, n_mono, n_samples) # where you store the generated data
    ising_energies = zeros(n_samples)
    
    ########################################################### Initialize some quantitites in this section
    polymers = Array{mp.Magnetic_polymer}(undef, n_temps)
    trajs = Array{mp.MC_data}(undef, n_temps)
    for i_temp in 1:n_temps
        polymers[i_temp] = mp.Magnetic_polymer(n_mono)#, inv_temps[i_temp], spins_coupling, alpha)
        trajs[i_temp] = mp.MC_data(n_strides*stride)
        #if isfile("simulation_data/final_config_$(n_mono).txt")
        #    mp.initialize_poly!(polymers[i_temp],"simulation_data/final_config_$(n_mono).txt")
        #else
        mp.initialize_poly!(polymers[i_temp])
        #end
    end 

    polymers_2 = Array{mp.Magnetic_polymer}(undef, n_temps)
    trajs_2 = Array{mp.MC_data}(undef, n_temps)
    println("ok, different polymers")
    for i_temp in 1:n_temps
        polymers_2[i_temp] = mp.Magnetic_polymer(n_mono)#, inv_temps[i_temp], spins_coupling, alpha)
        trajs_2[i_temp] = mp.MC_data(n_strides*stride)
        #if isfile("simulation_data/final_config_$(n_mono).txt")
        #    mp.initialize_poly!(polymers_2[i_temp],"simulation_data/final_config_$(n_mono).txt")
        #else
        println("scratch initialization")
        mp.initialize_poly!(polymers_2[i_temp])
        #end
    end 

    
    param_series = zeros(n_feats+1,n_params)
    #avg_spins = zeros(n_mono, n_samples)
    overlaps = zeros(n_feats+1, n_samples)
    avg_overlaps = zeros(n_feats+1)
    cov_overlaps = zeros(n_feats+1, n_feats+1)
    
    theta = zeros(n_feats+1)
    trial_theta = zeros(n_feats+1)
    theta .= initial_theta
    if theta[n_feats+1] <= 0 
        theta[n_feats+1] = 0.5
    end

    fields = zeros(n_mono)
    for i in 1:n_feats
        fields .+= features[:,i] .* theta[i]
    end
    #############################################################
    
    mp.set_spins!(polymers_2, data_spins[1,:])
    for i_eq in 1:500
        mp.MMC_run!(polymers,  trajs,  n_strides,stride,inv_temps, theta[n_feats+1], fields)  #theta[n_feats+1] is the spin coupling
        mp.MMC_run!(polymers_2,trajs_2,n_strides,stride,inv_temps, theta[n_feats+1], fields, true)  #theta[n_feats+1] is the spin coupling
        if i_eq%10 == 0
            println(i_eq)
        end
    end
    
    param_series[:,1] .= theta
    acceptance = 0.0

    energy = 0
    for i_data in 1:n_data
        for i_mono in 1:n_mono
            energy -= fields[i_mono]*data_spins[i_data,i_mono]
        end
    end

    energy_correction = 0.0

    ##
    theta_variation = zeros(n_feats+1)
    ##

    for i_param in 2:n_params
        
        #theta_variation .= (2 .* rand(n_feats+1) .-1) .*delta_theta
        theta_variation .= randn(n_feats+1) .* delta_theta
        trial_theta .= theta .+ theta_variation
        
        if trial_theta[n_feats+1] > 0 #&& trial_theta[n_feats+1] <= 1.0
            trial_fields = zeros(n_mono)
            for i in 1:n_feats
                trial_fields .+= features[:,i] .* trial_theta[i]
            end
            
            trial_energy = 0
            for i_data in 1:n_data
                for i_mono in 1:n_mono
                    trial_energy -= trial_fields[i_mono]*data_spins[i_data,i_mono]
                end
            end
            

            mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps, theta[n_feats+1], fields) # This is an short equilibration run so that the expectations are a bit better when changing weights
            mp.MMC_run!(polymers,trajs,n_strides,stride,inv_temps,spins_configs,ising_energies, sample_lag,n_samples, theta[n_feats+1], fields)
            
            
            fill!(overlaps, 0.0)
            for i_sample in 1:n_samples
                for i_feat in 1:n_feats
                    for i_mono in 1:n_mono
                        overlaps[i_feat, i_sample] += spins_configs[i_mono, i_sample]*features[i_mono, i_feat]
                    end
                end
                overlaps[n_feats+1, i_sample] = ising_energies[i_sample]
            end

            #overlaps = features'*spins_configs ## shorter but maybe less efficient way to write stuff in the previous nested for loops 
            avg_overlaps = vec(mean(overlaps, dims=2))
            cov_overlaps = cov(overlaps, dims=2)
            
            energy_correction = 0
            #linear correction
            for j in 1:n_feats+1
                energy_correction += theta_variation[j]*avg_overlaps[j]
            end
            #quadratic correction
            quadratic_correction = 0
            for j1 in 1:n_feats+1
                for j2 in 1:n_feats+1
                    quadratic_correction += theta_variation[j1]*cov_overlaps[j1,j2]*theta_variation[j2] * 0.5
                end
            end
            energy_correction += quadratic_correction
            energy_correction = energy_correction * n_data

            ##################################################
            
            energy_correction_J = 0.0
            for i_data in 1:n_data
                mp.set_spins!(polymers_2, data_spins[i_data,:])
                # here I run the MC's with the extra 'true' argument to quench the spins
                mp.MMC_run!(polymers_2,trajs_2,n_strides,stride,inv_temps, theta[n_feats+1], fields, true) # This is an short equilibration run so that the expectations are a bit better when changing weights
                mp.MMC_run!(polymers_2,trajs_2,n_strides,stride,inv_temps,spins_configs,ising_energies,sample_lag,n_samples, theta[n_feats+1], fields, true)
                energy_correction_J += mean(ising_energies) * theta_variation[n_feats+1] + 0.5 * theta_variation[n_feats+1]^2 * var(ising_energies)
            end

            
            ##################################################
            
            delta_acc = -(trial_energy - energy) - energy_correction + energy_correction_J
            delta_acc>=0 ? acceptance=1 : acceptance=exp(delta_acc)
            if i_param%10 == 0
                println(i_param)
                for i_feat in 1:n_feats
                    println("w$(i_feat): ",theta[i_feat]," ---> ",trial_theta[i_feat])
                end
                println("J : ",theta[n_feats+1]," ---> ",trial_theta[n_feats+1])
                println("delta_energy: ", -(trial_energy-energy))
                println("delta_acc: ", delta_acc)
                println("order of error: ", sum(theta_variation.^2)^1.5 *n_data)
                println("ratio quadratic_correction/tot_corr: ", quadratic_correction/energy_correction)
                println("ratio ising_correction/normal_correction: ", energy_correction_J/energy_correction)
                println("acceptance: ", acceptance)
                println("avg acceptance: ", accepted_moves/i_param)
            end
            if acceptance > rand()
                theta .= trial_theta
                energy = trial_energy
                fields .= trial_fields
                accepted_moves += 1
            end
        end


        for i_feat in 1:n_feats
            param_series[i_feat, i_param] = theta[i_feat]
        end
        param_series[n_feats+1, i_param] = theta[n_feats+1]

        if i_param%1000 == 0
            !isdir("AMHI_data") && mkdir("AMHI_data")
            open("AMHI_data/thetas_$(n_data)Qamh$(n_samples)_$(delta_theta).txt", "w") do io
                writedlm(io, param_series)
            end
        end
    end
    println("Acceptance ratio: ", accepted_moves/n_params)

 
    !isdir("AMHI_data") && mkdir("AMHI_data")
    open("AMHI_data/thetas_$(n_data)Qamh$(n_samples)_$(delta_theta).txt", "w") do io
        writedlm(io, param_series)
    end
    
    mp.write_results(polymers[1],trajs[1])

end


#####################################################################################
#####################################################################################

function generate_data(n_data::Int, features_file::String, n_strides::Int, weights::Array{Float64}, spins_coupling::Float64)
    io = open(features_file, "r")
    features = readdlm(io,Float64)
    close(io)
    n_mono = size(features, 1)
    n_feats = size(features, 2)

    #spins_coupling = 1.25
    println(spins_coupling, weights)
    #inv_temps = [10000.0, 0.9, 0.8, 0.74, 0.65, 0.62, 0.6, 0.55, 0.5, 0.4]
    #inv_temps = [0.01, 0.01]
    inv_temps = [1.0, 0.96, 0.92, 0.89, 0.85, 0.83, 0.82, 0.79, 0.75, 0.67, 0.57, 0.5, 0.4]
    n_temps = length(inv_temps)
    stride = 100

    polymers = Array{mp.Magnetic_polymer}(undef, n_temps)
    trajs = Array{mp.MC_data}(undef, n_temps)
    for i_temp in 1:n_temps
        polymers[i_temp] = mp.Magnetic_polymer(n_mono)#, inv_temps[i_temp], spins_coupling, alpha)
        trajs[i_temp] = mp.MC_data(n_strides*stride)
        #if isfile("simulation_data/final_config_$(n_mono).txt")
        #    mp.initialize_poly!(polymers[i_temp],"simulation_data/final_config_$(n_mono).txt")
        #    print("yes")
        #else
        mp.initialize_poly!(polymers[i_temp])
        #end
    end 

    fields = zeros(n_mono)
    for i in 1:n_feats
        fields .+= features[:,i] .* weights[i]
    end
    #mp.set_fields!(polymers, fields)
    
    #here I run the simulation for a while in order to equilibrate the chain
    for i_burnin in 1:50
        println("burnin number: ", i_burnin)
        mp.MMC_run!(polymers, trajs, n_strides, stride, inv_temps, spins_coupling, fields)
    end
    
    #n_data = 500
    summary_stats = zeros(n_feats,n_data)
    spins_conf = zeros(Int,n_mono,n_data)
    

    
    poly_confs = zeros(Int,n_data*3,n_mono)
    

    for i_data in 1:n_data
        println("Number of data point:  ", i_data)
        mp.MMC_run!(polymers, trajs, n_strides, stride, inv_temps, spins_coupling, fields)
        #mp.write_results(polymers[1],trajs[1])
        spins_conf[:,i_data] .= polymers[1].spins
        for j in 1:3
            poly_confs[(i_data-1)*3+j,:] .= polymers[1].coord[j,:]
        end
        if i_data%10 == 0
            open("saw_conf_data_prova.txt","w") do io
                writedlm(io,transpose([weights; spins_coupling]))
                writedlm(io,transpose(spins_conf))
            end
            open("poly_confs_prova.txt","w") do io
                writedlm(io,transpose([weights; spins_coupling]))
                writedlm(io, poly_confs)
            end
        end
    end
    #compute_summary_stats!(summary_stats,spins_conf,features)
    #println(summary_stats)
    #println(spins_conf)
    open("saw_conf_data_prova.txt","w") do io
        writedlm(io,transpose([weights; spins_coupling]))
        writedlm(io,transpose(spins_conf))
    end
    #=open("data_file.txt","w") do io
        writedlm(io,transpose(weights))
        writedlm(io,transpose(summary_stats))
    end=#
    open("poly_confs_prova.txt","w") do io
        writedlm(io,transpose([weights; spins_coupling]))
        writedlm(io, poly_confs)
    end


    mp.write_results(polymers[1])
end

## The previous function generates samples from the likelihood with a fixed set of parameters e.g. MAP estimates
## This one generates samples from the posterior predictive or bayesian predictive distribution

function generate_data_posterior_predictive(features_file::String, n_strides::Int, n_data_per_param::Int, param_file::String)
    io = open(features_file, "r")
    features = readdlm(io,Float64)
    close(io)
    n_mono = size(features, 1)
    n_feats = size(features, 2)

    io = open(param_file, "r")
    post_samples = readdlm(io, Float64)
    close(io)
    n_param = size(post_samples, 2)
    println(post_samples[:,1])
    

    inv_temps = [1.0, 0.96, 0.92, 0.89, 0.85, 0.83, 0.82, 0.79, 0.75, 0.67, 0.57, 0.5, 0.4]
    n_temps = length(inv_temps)
    stride = 100

    n_data = n_data_per_param * n_param

    spins_conf = zeros(Int,n_mono,n_data) 
    poly_confs = zeros(Int,n_data*3,n_mono)

    polymers = Array{mp.Magnetic_polymer}(undef, n_temps)
    trajs = Array{mp.MC_data}(undef, n_temps)
    for i_temp in 1:n_temps
        polymers[i_temp] = mp.Magnetic_polymer(n_mono)#, inv_temps[i_temp], spins_coupling, alpha)
        trajs[i_temp] = mp.MC_data(n_strides*stride)
        mp.initialize_poly!(polymers[i_temp])
    end 

    ###################################################################

    fields = zeros(n_mono)
    for i in 1:n_feats
        fields .+= features[:,i] .* post_samples[i,1]
    end
    #mp.set_fields!(polymers, fields)
    
    #here I run the simulation for a while in order to equilibrate the chain
    for i_burnin in 1:15
        println("burnin number: ", i_burnin)
        mp.MMC_run!(polymers, trajs, n_strides, stride, inv_temps, post_samples[n_feats+1,1], fields)
    end

    ###################################################################

    for i_param in 1:n_param
        println("I_PARAM: ", i_param)
        println(post_samples[:,i_param])
        fields = zeros(n_mono)
        for i in 1:n_feats
            fields .+= features[:,i] .* post_samples[i,i_param]
        end
        # when changing parameters short equilibration run
        for i_burnin in 1:5
            println("burnin number: ", i_burnin)
            mp.MMC_run!(polymers, trajs, n_strides, stride, inv_temps, post_samples[n_feats+1,i_param], fields)
        end

        for i_data in 1:n_data_per_param
            println(i_data)
            mp.MMC_run!(polymers, trajs, n_strides, stride, inv_temps, post_samples[n_feats+1,i_param], fields)
            #mp.write_results(polymers[1],trajs[1])
            spins_conf[:,i_data + (i_param-1)*n_data_per_param] .= polymers[1].spins
            for j in 1:3
                poly_confs[(i_data + (i_param-1)*n_data_per_param-1)*3+j,:] .= polymers[1].coord[j,:]
            end
        end
    end

    open("saw_conf_$(param_file)","w") do io
        #writedlm(io,transpose([weights; spins_coupling]))
        writedlm(io,transpose(spins_conf))
    end
    
    open("poly_confs_$(param_file)","w") do io
        #writedlm(io,transpose([weights; spins_coupling]))
        writedlm(io, poly_confs)
    end


end

