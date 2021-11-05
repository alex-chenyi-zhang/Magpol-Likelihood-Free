using Random, Distributions, DelimitedFiles, CSV
Random.seed!()

# Here I define the global quantities used by all pivot MCMC run which are the pivot moves
const pivot_moves = Array{Int,3}(zeros(47, 2, 3)) # Tensor that defines all the pivot moves of the octahedral symmetry group
const perms = [[1 2 3]; [1 3 2]; [2 1 3]; [2 3 1]; [3 1 2]; [3 2 1]]
const tr_signs = [[1 1 1]; [1 1 -1]; [1 -1 1]; [1 -1 -1]; [-1 1 1]; [-1 1 -1]; [-1 -1 1]; [-1 -1 -1]]
p_moves_count = 1
for i in 1:6
    for j in 1:8
        if i==1 && j==1
            continue
        else
            pivot_moves[p_moves_count, 1, :] .= perms[i,:]
            pivot_moves[p_moves_count, 2, :] .= tr_signs[j,:]
            
            global p_moves_count += 1
        end
    end
end

#########################################################################
#########################################################################

# This structure contains all I need to compute the Hamiltonian
struct Magnetic_polymer{T1<:Int, T2<:AbstractFloat}
    n_mono::T1
    inv_temp::T2
    J::T2
    alpha::T2
    energy::T2
    spins::Array{T1}
    fields::Array{T2}
    coord::Array{T1,2}         # Coordinates of the monomers
    trial_coord::Array{T1,2}
    hash_saw::Dict{Int64, Int64}
    neighbours::Array{T1, 2}
    trial_neighbours::Array{T1, 2}
end

function Magnetic_polymer(n_mono::T1, inv_temp::T2, J::T2, alpha::T2) where {T1<:Int, T2<:AbstractFloat}
    Magnetic_polymer(n_mono, inv_temp, J, alpha, 0.0, zeros(Int, n_mono), zeros(n_mono), zeros(Int, 3,n_mono),
    zeros(Int, 3,n_mono), Dict{Int, Int}(), zeros(Int, 7,n_mono), zeros(Int, 7,n_mono))
end

function initialize_poly!(poly::Magnetic_polymer)
    poly.spins .= rand((-1,1), poly.n_mono)
    #poly.fields .= (rand(poly.n_mono).*2) .-1
    for i_mono in 1:poly.n_mono
        poly.coord[1, i_mono] = i_mono-1
    end    
    for i_mono in 1:poly.n_mono
        for j in 1:3
            poly.trial_coord[j, i_mono] = poly.coord[j, i_mono]
        end
    end
end

#The following method of the "initialize_poly!()" function allows initialization from file
function initialize_poly!(poly::Magnetic_polymer, file_name::String)
    io = open(file_name, "r")
    data = readdlm(io,Int64)
    close(io)
    for i_mono in 1:poly.n_mono
        for j in 1:3
            poly.coord[j,i_mono] = data[i_mono,j]
        end
        poly.spins[i_mono] = data[i_mono, 4]
    end
end


function display_polymer(polymer::Magnetic_polymer)
    println("n_mono: $(polymer.n_mono)")
    println("inv_temp: $(polymer.inv_temp)")
    println("coupling: $(polymer.J)")
    #=for i in 1:(polymer.n_mono)
        println(polymer.coord[:,i])
    end
    for i in 1:(polymer.n_mono)
        println("f$i: $(polymer.fields[i])")
    end
    for i in 1:(polymer.n_mono)
        println("s$i: $(polymer.spins[i])")
    end
    println(typeof(polymer.n_mono))
    println(typeof(polymer.inv_temp))
    println(typeof(polymer.J))
    println(typeof(polymer.alpha))
    polymer.spins=#
    println(polymer.hash_saw)
end

# This structure stores series of observables I'm computing along the MCMC
struct MC_data{T1<:Int, T2<:AbstractFloat}
    n_steps::T1
    energies::Array{T2}
    magnetization::Array{T2}
    rg2::Array{T2}
end

function MC_data(n_steps::Int)
    MC_data(n_steps, zeros(n_steps), zeros(n_steps), zeros(n_steps))
end

#########################################################################
#########################################################################


function try_pivot!(k::Int, move::Int, coo::Array{Int,2}, t_coo::Array{Int,2}, n_mono::Int)
    for i_mono in 1:k
        for j in 1:3
            #pol.trial_coord[j, i_mono] = pol.coord[j, i_mono]
            t_coo[j, i_mono] = coo[j, i_mono]
        end
    end
    for i_mono in k+1:n_mono
        for j in 1:3
            t_coo[j,i_mono] = (pivot_moves[move,2,j]*(coo[pivot_moves[move,1,j],i_mono]
                                         - coo[pivot_moves[move,1,j],k]) + coo[j,k])
        end
    end
end

#########################################################################

function checksaw!(k::Int, pol::Magnetic_polymer, a::Int)
    L = max(k, pol.n_mono-k) + 1
    is_saw = true
    t = 0
    while(t<L && is_saw)
        if k+t <= pol.n_mono
            saw_key = pol.trial_coord[1,k+t]*a^2 + pol.trial_coord[2,k+t]*a + pol.trial_coord[3,k+t]
            if haskey(pol.hash_saw, saw_key)
                is_saw = false
            else
                pol.hash_saw[saw_key] = k+t
            end
        end
        if k-t >= 1 && t!=0
            saw_key = pol.trial_coord[1,k-t]*a^2 + pol.trial_coord[2,k-t]*a + pol.trial_coord[3,k-t]
            if haskey(pol.hash_saw, saw_key)
                is_saw = false
            else
                pol.hash_saw[saw_key] = k-t
            end
        end
        t += 1
    end
    return is_saw
end

#########################################################################
## poi quando si faranno le mosse locali non si userà questa funnzione perché 
## non ci sarà bisogno di ricalcolare TUTTA la lista dei vicini
function compute_neighbours!(coo::Array{Int,2}, near::Array{Int,2}, dic::Dict{Int,Int}, a::Int, n_mono::Int)
    for i_mono in 1:n_mono
        near[1, i_mono] = 0
        for j in 2:7
            near[j, i_mono] = -1
        end
        saw_key_base = coo[1,i_mono]*a^2 + coo[2,i_mono]*a + coo[3,i_mono]
        for j in 1:3
            for k in -1:2:1
                #coo[j,i_mono] += k
                #saw_key = coo[1,i_mono]*a^2 + coo[2,i_mono]*a + coo[3,i_mono]
                saw_key = saw_key_base + k*a^(3-j)
                if haskey(dic, saw_key)
                    near[1, i_mono] += 1
                    n_neigh = near[1, i_mono]+1
                    near[n_neigh, i_mono] = dic[saw_key]
                end
                #coo[j,i_mono] -= k
            end
        end
    end
end

#########################################################################

function gyration_radius_squared(pol::Magnetic_polymer)
    rcm = zeros(3)
    rg2 = 0.0
    for i in 1:pol.n_mono
        for j in 1:3
            #rg2 += sum((view(pol.coord,:,i) .- rcm).^2)
            rg2 += pol.coord[j,i]^2
            rcm[j] += pol.coord[j,i]
        end
    end
    r = 0.0
    for j in 1:3
        r += rcm[j]^2
    end
    return rg2/pol.n_mono - r/(pol.n_mono)^2
end

#########################################################################

function compute_new_energy(pol::Magnetic_polymer, near::Array{Int,2}, ff::Array{Float64}, s::Array{Int})
    ene = 0.0
    ene_J = 0.0
    for i_mono in 1:pol.n_mono
        ene -= ff[i_mono] * s[i_mono]
        for j in 1:near[1, i_mono]
            ene_J -= s[i_mono] * s[near[j+1,i_mono]] * pol.J * 0.5
        end
    end
    return ene*(1-pol.alpha) + ene_J*pol.alpha
end


#########################################################################
#########################################################################
### In this section I there are the functions to perform local moves on the self-avoiding walk

function single_bead_flip!(t_coo::Array{Int,2}, dic::Dict{Int, Int}, n_mono::Int, a::Int)
    mono = rand(2:n_mono-1)
    dist = 0
    for j in 1:3
        dist += (t_coo[j,mono+1]-t_coo[j,mono-1])^2
    end
    if dist == 2
        saw_key = 0
        for j in 1:3
            saw_key += (t_coo[j,mono-1]+t_coo[j,mono+1]-t_coo[j,mono])*a^(3-j)
        end
        
        if !haskey(dic,saw_key)
            delete!(dic, t_coo[1,mono]*a^2 + t_coo[2,mono]*a + t_coo[3,mono])
            for j in 1:3
                t_coo[j,mono] = t_coo[j,mono-1]+t_coo[j,mono+1]-t_coo[j,mono]
            end
            dic[saw_key] = mono
        end
    end
end

#########################################################################

function crankshaft_180!(t_coo::Array{Int,2}, dic::Dict{Int, Int}, n_mono::Int, a::Int)
    mono = rand(1:n_mono-3)
    dist = 0
    for j in 1:3
        dist += (t_coo[j,mono+3]-t_coo[j,mono])^2
    end
    if dist==1
        saw_key1 = 0
        saw_key2 = 0
        for j in 1:3
            saw_key1 += (2*t_coo[j,mono] - t_coo[j,mono+1])*a^(3-j)
            saw_key2 += (2*t_coo[j,mono+3] - t_coo[j,mono+2])*a^(3-j)
        end
        if !haskey(dic,saw_key1) && !haskey(dic,saw_key2)
            delete!(dic, t_coo[1,mono+1]*a^2 + t_coo[2,mono+1]*a + t_coo[3,mono+1])
            delete!(dic, t_coo[1,mono+2]*a^2 + t_coo[2,mono+2]*a + t_coo[3,mono+2])
            for j in 1:3
                t_coo[j,mono+1] = 2*t_coo[j,mono] - t_coo[j,mono+1]
                t_coo[j,mono+2] = 2*t_coo[j,mono+3] - t_coo[j,mono+2]
            end
            dic[saw_key1] = mono+1
            dic[saw_key2] = mono+2
        end
    end
end

#########################################################################

function crankshaft_90_270!(t_coo::Array{Int,2}, dic::Dict{Int, Int}, n_mono::Int, a::Int, new_coord1::Array{Int}, new_coord2::Array{Int})
    mono = rand(1:n_mono-3)
    dist = 0
    axis = 0
    for j in 1:3
        dist += (t_coo[j,mono+3]-t_coo[j,mono])^2
    end
    if dist == 1
        orient = 0
        for j in 1:3
            if (t_coo[j,mono+3]-t_coo[j,mono]) != 0
                axis = j
                orient = t_coo[j,mono+3]-t_coo[j,mono]
                break
            end
        end
        p=0
        t=0
        if axis==1
            p=2
            t=rand((3,2))
        elseif axis==2
            p=6
            t=rand((2,5))
        elseif axis==3
            p=3
            t=rand((5,3))
        else
            println("Invalid Rotation Axis!")
        end
        #new_coord1 = zeros(Int,3)
        #new_coord2 = zeros(Int,3)
        for j in 1:3 
            new_coord1[j] = t_coo[j,mono] + tr_signs[t,j]*(t_coo[perms[p,j],mono+1]-t_coo[perms[p,j],mono])*orient
            new_coord2[j] = t_coo[j,mono+3] + tr_signs[t,j]*(t_coo[perms[p,j],mono+2]-t_coo[perms[p,j],mono+3])*orient
        end
        
        saw_key1 = new_coord1[1]*a^2 + new_coord1[2]*a + new_coord1[3]
        saw_key2 = new_coord2[1]*a^2 + new_coord2[2]*a + new_coord2[3]
        if !haskey(dic,saw_key1) && !haskey(dic,saw_key2)
            delete!(dic, t_coo[1,mono+1]*a^2 + t_coo[2,mono+1]*a + t_coo[3,mono+1])
            delete!(dic, t_coo[1,mono+2]*a^2 + t_coo[2,mono+2]*a + t_coo[3,mono+2])
            for j in 1:3
                t_coo[j,mono+1] = new_coord1[j]
                t_coo[j,mono+2] = new_coord2[j]
            end
            dic[saw_key1] = mono+1
            dic[saw_key2] = mono+2
        end
    end 
end


#########################################################################
#########################################################################
function spins_MC!(pol::Magnetic_polymer, n_flips::Int, ff::Array{Float64}, s::Array{Int}, near::Array{Int,2})
    delta_tot = 0.0
    for i_flip in 1:n_flips
        flip_candidate = rand(1:pol.n_mono)
        local_ene = 0.0
        local_ene -= ff[flip_candidate] * s[flip_candidate] * (1-pol.alpha)
        for j in 1:near[1,flip_candidate]
            local_ene -= pol.J * pol.alpha * s[flip_candidate] * s[near[j+1,flip_candidate]]
        end

        s[flip_candidate]==1 ? trial_spin_value=-1 : trial_spin_value=1

        trial_local_ene = 0.0
        trial_local_ene -= ff[flip_candidate] * trial_spin_value * (1-pol.alpha)
        for j in 1:near[1,flip_candidate]
            trial_local_ene -= pol.J * pol.alpha * trial_spin_value * s[near[j+1,flip_candidate]]
        end

        delta_ene = trial_local_ene - local_ene
        acc = 0.0

        delta_ene<=0 ? acc=1.0 : acc=exp(-delta_ene*pol.inv_temp)

        if acc > rand()
            delta_tot += delta_ene
            s[flip_candidate] = trial_spin_value
        end
    end
    return delta_tot
end


#########################################################################
#########################################################################

function MC_run!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int)
    for i_mono in 1:pol.n_mono
        for j in 1:3
            pol.trial_coord[j,i_mono] = pol.coord[j,i_mono]
        end
    end 
    new_coord1 = zeros(Int,3)
    new_coord2 = zeros(Int,3)
    n_acc = 0
    n_pivots = 0
    #=
      I represent the possible positions of the monomers in the cubic lattice
      as seeing the vector of coordinates as the representation in base "hash_base".
      e.g. (xi,yi,zi) --> xi*hash_base^2 + yi*hash_base + zi*1
      Since the polymer is fixed in the origin the coordinates are ≤ n_mono in modulus, so if I take hash_base 
      >= 2*n_mono + 1, the mapping I think it should be 1-to-1. This speeds up the code significantly as the 
      hashing of integer valued keys for the dictionary containing a certain polymer config. is much 
      cheaper than hashing keys which are arrays sliced from a matrix 
    =#
    hash_base = 2*pol.n_mono + 1

    checksaw!(1, pol, hash_base)
    compute_neighbours!(pol.coord, pol.neighbours, pol.hash_saw, hash_base, pol.n_mono)
    energy = compute_new_energy(pol,pol.neighbours,pol.fields, pol.spins)

    traj.energies[start] = energy
    traj.magnetization[start] = mean(pol.spins)
    traj.rg2[start] = gyration_radius_squared(pol)

    empty!(pol.hash_saw)
    
    for i_step in (start+1):finish
        pivot = rand(2:pol.n_mono-1)
        p_move = rand(1:47)
        if i_step%10000 == 0
            println("Pivot: ", pivot)
            println("Attmpted move: ", p_move)
            println("step number: ", i_step, "\n")
        end
        try_pivot!(pivot, p_move, pol.coord, pol.trial_coord, pol.n_mono)
        still_saw = checksaw!(pivot, pol, hash_base)


        # If the pivot move was unsuccessful try next move on the previous configuration
        if !still_saw
            empty!(pol.hash_saw)
            for i_mono in 1:pol.n_mono
                for j in 1:3
                    pol.trial_coord[j,i_mono] = pol.coord[j,i_mono]
                end
                pol.hash_saw[pol.trial_coord[1,i_mono]*hash_base^2+pol.trial_coord[2,i_mono]*hash_base+pol.trial_coord[3,i_mono]] = i_mono
            end
        else
            n_pivots += 1
        end


        for i_local in 1:pol.n_mono
            mv = rand(1:4)
            if mv==1
                single_bead_flip!(pol.trial_coord,pol.hash_saw,pol.n_mono,hash_base)
            elseif mv==2
                crankshaft_180!(pol.trial_coord,pol.hash_saw,pol.n_mono,hash_base)
            else
                crankshaft_90_270!(pol.trial_coord,pol.hash_saw,pol.n_mono,hash_base,new_coord1,new_coord2)
            end
        end
        
        compute_neighbours!(pol.trial_coord, pol.trial_neighbours, pol.hash_saw, hash_base, pol.n_mono)
        trial_energy = compute_new_energy(pol,pol.trial_neighbours,pol.fields, pol.spins)
        delta_energy = trial_energy - energy
        acc = 0.0
        delta_energy<=0 ? acc=1.0 : acc=exp(-delta_energy*pol.inv_temp)
        if acc > rand()
            for i_mono in 1:pol.n_mono
                for j in 1:3
                    pol.coord[j,i_mono] = pol.trial_coord[j,i_mono]
                end
            end
            for i_mono in 1:pol.n_mono
                for j in 1:7
                    pol.neighbours[j,i_mono] = pol.trial_neighbours[j,i_mono]
                end
            end
            energy = trial_energy
            n_acc += 1
        end
        
        #pre_ene = energy
        #spins_delta = spins_MC!(pol, pol.n_mono,pol.fields,pol.spins,pol.neighbours) ## spins_MC! return the total energy variation of the accepted spin n_flips
        #energy = compute_new_energy(pol,pol.neighbours,pol.fields,pol.spins)
        
        energy += spins_MC!(pol, pol.n_mono,pol.fields,pol.spins,pol.neighbours) ## spins_MC! return the total energy variation of the accepted spin n_flips
        traj.rg2[i_step] = gyration_radius_squared(pol)
        traj.magnetization[i_step] = mean(pol.spins)
        traj.energies[i_step] = energy

        # This next if ... end can be removed. I kept as a sanity check in an early stage 
        # when I had problems with the computation of the nearest neighbours
        if !isinteger(energy)
            println("Non integer energy:  ", energy)
            println(pol.coord)
            println(pol.neighbours .- pol.trial_neighbours)
            println(pol.trial_neighbours)
            break
        end

        #println(energy-compute_new_energy(pol,pol.neighbours,pol.fields,pol.spins))
        #println(energy-pre_ene - spins_delta)

        empty!(pol.hash_saw)
    end
    #println("Fraction of accepted moves: ", n_acc/(finish-1))
    #println("Fraction of successful pivots: ", n_pivots/(finish-1))
end

function MC_run!(pol::Magnetic_polymer, traj::MC_data)
    MC_run!(pol, traj, 1, traj.n_steps)
end


function write_results(pol::Magnetic_polymer, traj::MC_data)
    !isdir("simulation_data") && mkdir("simulation_data")
    
    open("simulation_data/final_config_$(pol.n_mono)_$(pol.inv_temp).txt", "w") do io
        writedlm(io, [transpose(pol.coord) pol.spins])
    end
    open("simulation_data/MC_data_$(pol.n_mono)_$(pol.inv_temp).txt", "w") do io
        writedlm(io, [traj.energies traj.magnetization traj.rg2])
    end
end

function write_results(pol::Magnetic_polymer)
    !isdir("simulation_data") && mkdir("simulation_data")

    open("simulation_data/final_config_$(pol.n_mono)_$(pol.inv_temp).txt", "w") do io
        writedlm(io, [transpose(pol.coord) pol.spins])
    end
end

#########################################################################
#########################################################################

function simulation(n_mono::Int, n_steps::Int ,beta_temp::Float64, spins_coupling::Float64, alpha::Float64)
    simulation_data = MC_data(n_steps)
    polymer = Magnetic_polymer(n_mono, beta_temp, spins_coupling, alpha)
    initialize_poly!(polymer)
    MC_run!(polymer, simulation_data)
    write_results(polymer, simulation_data)
end

function simulation(n_mono::Int, n_steps::Int ,beta_temp::Float64, spins_coupling::Float64, alpha::Float64, conf_file::String)
    simulation_data = MC_data(n_steps)
    polymer = Magnetic_polymer(n_mono, beta_temp, spins_coupling, alpha)
    initialize_poly!(polymer, conf_file)
    MC_run!(polymer, simulation_data)
    write_results(polymer, simulation_data)
end



function simulation(input::String)
    # First read parameters from input
    io = open("input.txt", "r")
    n_mono = 1
    n_steps = 1
    beta_temp = 1.0
    spins_coupling = 0.0
    alpha = 1.0
    conf_file = ""
    from_file = false
    while ! eof(io)
        line = split(readline(io))
        if line[1] == "n_mono"
            n_mono = parse(Int, line[2])
        elseif line[1] == "n_steps"
            n_steps = parse(Int, line[2])
        elseif line[1] == "spins_coupling"
            spins_coupling = parse(Float64, line[2])
        elseif line[1] == "alpha"
            alpha = parse(Float64, line[2])
        elseif line[1] == "inverse_temperature"
            beta_temp = parse(Float64, line[2])
        elseif line[1] == "initialize_from_files"
            from_file = parse(Bool, line[2])
        elseif line[1] == "conf_file"
            conf_file = String(line[2])
        end
    end
    
    close(io)

    simulation_data = MC_data(n_steps)
    polymer = Magnetic_polymer(n_mono, beta_temp, spins_coupling, alpha)
    
    if from_file
        initialize_poly!(polymer, conf_file)
    else
        initialize_poly!(polymer)
    end
    
    MC_run!(polymer, simulation_data)
    write_results(polymer, simulation_data)
end
