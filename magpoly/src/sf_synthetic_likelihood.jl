include("magpoly.jl")

using .magpoly
using DelimitedFiles, Random
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


## for now this is just the backbone of the function to carry out the likelihood-free inference task
function synthetic_likelihood_polymer(n_samples::Int, n_params::Int, sample_lag::Int, features_file::String)
    io = open(features_file, "r")
    features = readdlm(io,Float64)
    close(io)
    n_mono = length(features)

    stride = 500
    n_steps = n_samples*sample_lag
    n_strides = div(n_steps, stride)
    spins_configs = zeros(Int, n_mono, n_samples)
    spins_coupling = 1.0
    alpha = 0.5
    inv_temps = [1.0, 0.9, 0.8, 0.74, 0.65, 0.62, 0.6, 0.57, 0.55, 0.5, 0.4]
    n_temps = length(inv_temps)
    
    
    polymers = Array{Magnetic_polymer}(undef, n_temps)
    trajs = Array{MC_data}(undef, n_temps)
    for i_temp in 1:n_temps
        polymers[i_temp] = Magnetic_polymer(n_mono, inv_temps[i_temp], spins_coupling, alpha)
        trajs[i_temp] = MC_data(n_strides*stride)
        initialize_poly!(polymers[i_temp])#,"simulation_data/final_config_$(n_mono)_$(inv_temps[i_temp]).txt")
    end 


    for i_param in 1:n_params
        # I think basically here I just need to recompute the external fields using the new parameters vector
        # and the features. I set these fields with the initialize_fields!() function and then I'm good to run the 
        # new MMC simulation

        ## Of course then I need all the part about how to make the move in parameters space and how to acc/rej it with tynthetic likelihood
        MMC_run!(polymers,trajs,n_strides,stride,inv_temps,spins_configs,sample_lag)
    end
    for i_temp in 1:n_temps
        write_results(polymers[i_temp],trajs[i_temp])
    end
end

function prova_save()
    stride = 500
    n_samples = 20000
    sample_lag = 5
    n_steps = sample_lag*n_samples
    n_strides = div(n_steps,stride)
    spins_configs = zeros(Int, 200, n_samples)
    simulation_data = magpoly.MC_data(n_steps)
    polymer = magpoly.Magnetic_polymer(200, 1.0, 1.0, 1.0)
    magpoly.initialize_poly!(polymer)

    for i_strides in 1:n_strides
        magpoly.MC_run!(polymer, simulation_data,(i_strides-1)*stride+1,i_strides*stride, spins_configs,sample_lag)
    end
    println(spins_configs[:,1])
end