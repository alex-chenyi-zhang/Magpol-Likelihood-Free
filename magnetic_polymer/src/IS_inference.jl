include("magpoly.jl")

using .magpoly
const mp = magpoly
using DelimitedFiles, Random, Distributions, LinearAlgebra

function generate_saw(n_samples::Int, samples_lag::Int)
    io = open("features.txt", "r")
    features = readdlm(io,Float64)
    close(io)
    n_mono = size(features, 1)

    burn_in = ceil(Int, 20*n_mono^(1.11))
    #burn_in = 3
    pol = mp.Magnetic_polymer(n_mono, 1.0, 0.0, 1.0)
    mp.initialize_poly!(pol)

    for i_mono in 1:n_mono
        for j in 1:3
            pol.trial_coord[j,i_mono] = pol.coord[j,i_mono]
        end
    end

    hash_base = 2*n_mono +1

    # equilibration of the chain
    for i in 1:burn_in
        pivot = rand(2:n_mono-1)
        p_move = rand(1:47)
        mp.try_pivot!(pivot, p_move, pol.coord, pol.trial_coord, pol.n_mono)
        still_saw = mp.checksaw!(pivot, pol, hash_base)
        #println(still_saw)
        if still_saw
            for i_mono in 1:n_mono
                for j in 1:3
                    pol.coord[j,i_mono] = pol.trial_coord[j,i_mono]
                end
            end
        end
        empty!(pol.hash_saw)
    end
    println("End of burn in!")
    n_acc = 0

    n_steps = n_samples*samples_lag

    saw_configs = zeros(Int, n_samples, 3, n_mono)

    for i in 1:n_steps
        pivot = rand(2:n_mono-1)
        p_move = rand(1:47)
        if i%10000 == 0
            println("Pivot: ", pivot)
            println("Attmpted move: ", p_move)
            println("step number: ", i, "\n")
        end
        mp.try_pivot!(pivot, p_move, pol.coord, pol.trial_coord, pol.n_mono)
        still_saw = mp.checksaw!(pivot, pol, hash_base)
        if still_saw
            for i_mono in 1:n_mono
                for j in 1:3
                    pol.coord[j,i_mono] = pol.trial_coord[j,i_mono]
                end
            end
            n_acc += 1
        end

        if i%samples_lag == 0
            i_sample = div(i,samples_lag)
            for i_mono in 1:n_mono
                for j in 1:3
                    saw_configs[i_sample,j,i_mono] = pol.coord[j,i_mono]
                end
            end
        end
        empty!(pol.hash_saw)
    end
    println("Fraction of accepted pivots: ", n_acc/n_steps)

    !isdir("SAW_data") && mkdir("SAW_data")
    io = open("SAW_data/saw_configs.txt", "w")
    for i_sample in 1:n_samples
        writedlm(io, saw_configs[i_sample,:,:])
    end
    close(io)
end

# This function generates samples from the posterior distribution of the weights
# Assuming that the data were generated with zero spins coupling i.e. a logistic regression problem
# The sampling is done with a simple metroplis hastings sampler.
function vanilla_sampler(n_samples::Int, delta_w::Float64, initial_weights::Array{Float64}, features_file::String, data_file::String)
    io = open(features_file, "r")
    features = readdlm(io,Float64)#[1601:end,:]
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

    weights_series = zeros(n_feats, n_samples)
    energies = zeros(n_samples)

    weights = zeros(n_feats)
    trial_weights = zeros(n_feats)
    weights .= initial_weights
    weights_series[:,1] .= weights

    fields = zeros(n_mono)
    for i in 1:n_feats
        fields .+= features[:,i] .* weights[i]
    end

    energy = 0.0
    for i_data in 1:n_data
        for i_mono in 1:n_mono
            energy += data_spins[i_data, i_mono]*fields[i_mono] - log(1+exp(fields[i_mono]))
        end
    end
    energies[1] = energy

    w_acceptance = 0.0
    n_acc = 0

    for i_sample in 2:n_samples
        trial_weights .= weights .+ randn(n_feats) .* delta_w
        fields = zeros(n_mono)
        for i in 1:n_feats
            fields .+= features[:,i] .* trial_weights[i]
        end

        #trial_energy = -0.5*(trial_weights[1]^2 + trial_weights[2]^2)/(w_std^2)
        trial_energy = 0.0
        for i_data in 1:n_data
            for i_mono in 1:n_mono
                trial_energy += data_spins[i_data, i_mono]*fields[i_mono] - log(1+exp(fields[i_mono]))
            end
        end

        delta_energy = trial_energy - energy
        delta_energy >=0 ? w_acceptance=1 : w_acceptance=exp(delta_energy)


        if i_sample%100 == 0
            println(i_sample)
            for i_feat in 1:n_feats
                println("w$(i_feat): ",weights[i_feat]," ---> ",trial_weights[i_feat])
            end
            println("delta_energy: ", delta_energy)
            println("acceptance: ", w_acceptance)
        end

        if rand() <= w_acceptance
            energy = trial_energy
            weights .= trial_weights
            n_acc += 1
        end

        energies[i_sample] = energy
        weights_series[:, i_sample] .= weights
    end

    println("Acceptance fraction: ", n_acc/(n_samples-1))

    !isdir("IS_data") && mkdir("IS_data")
    open("IS_data/weights$(n_data)_$(delta_w)_uncoupled.txt", "w") do io
        writedlm(io, weights_series)
    end
    open("IS_data/neg_energies$(n_data)_$(delta_w)_uncoupled.txt", "w") do io
        writedlm(io, energies)
    end
end

function generate_spin_confs(features_file::String, n_data::Int, weights::Array{Float64})
    io = open(features_file, "r")
    features = readdlm(io,Float64)#[1601:end,:]
    close(io)
    n_mono = size(features, 1)
    n_feats = size(features, 2)
    println(n_mono)
    fields = zeros(n_mono)
    for i in 1:n_feats
        fields .+= features[:,i] .* weights[i]
    end
    proba = zeros(n_mono)
    for i in 1:n_mono
        proba[i] = exp(fields[i]) / (1 + exp(fields[i]))
    end
    spins_conf = zeros(Int, n_mono, n_data)
    for i_data in 1:n_data
        for i_mono in 1:n_mono
            if rand() <= proba[i_mono]
                spins_conf[i_mono, i_data] = 1
            end
        end
    end

    open("saw_conf_uncoupled.txt", "w") do io
        writedlm(io,transpose([weights]))
        writedlm(io,transpose(spins_conf))
    end
end

function LR_posterior_predictive(features_file::String, param_file::String)
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

    spins_conf = zeros(n_mono,n_param)

    for i_param in 1:n_param
        println("I_PARAM: ", i_param)
        println(post_samples[:,i_param])
        fields = zeros(n_mono)
        for i in 1:n_feats
            fields .+= features[:,i] .* post_samples[i,i_param]
        end
        proba = zeros(n_mono)
        for i in 1:n_mono
            proba[i] = exp(fields[i]) / (1 + exp(fields[i]))
        end
        spins_conf[:,i_param] .= proba
    end

    open("saw_conf_predR2_$(param_file)","w") do io
        #writedlm(io,transpose([weights; spins_coupling]))
        writedlm(io,transpose(spins_conf))
    end

end
