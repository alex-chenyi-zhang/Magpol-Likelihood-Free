using Random, Distributions, DelimitedFiles
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
struct Magnetic_polymer{T1<:Int}
    n_mono::T1
    spins::Array{T1}
    coord::Array{T1,2}         # Coordinates of the monomers
    trial_coord::Array{T1,2}
    hash_saw::Dict{Int64, Int64}
    neighbours::Array{T1, 2}
    trial_neighbours::Array{T1, 2}
end


function Magnetic_polymer(n_mono::Int64) 
    Magnetic_polymer(n_mono, zeros(Int, n_mono), zeros(Int, 3,n_mono),
    zeros(Int, 3,n_mono), Dict{Int, Int}(), zeros(Int, 7,n_mono), zeros(Int, 7,n_mono))
end

function initialize_poly!(poly::Magnetic_polymer, spin_values::Array{Int})
    poly.spins .= rand(spin_values, poly.n_mono)
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

function initialize_poly!(poly::Magnetic_polymer)
    initialize_poly!(poly,[0,1])
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




function set_spins!(polymers::Array{Magnetic_polymer}, spins::Array{Int64})
    for polymer in polymers
        for i_mono in 1:polymer.n_mono
            polymer.spins[i_mono] = spins[i_mono]
        end
    end
end
#=
function set_fields!(polymers::Array{Magnetic_polymer}, ff::Array{Float64})
    for polymer in polymers
        for i_mono in 1:polymer.n_mono
            polymer.fields[i_mono] = ff[i_mono]
        end
    end
end

function set_coupling!(polymers::Array{Magnetic_polymer}, spin_coupling::Float64)
    for polymer in polymers
        polymer.J = spin_coupling
    end
end=#

# This structure stores series of observables I'm computing along the MCMC
struct MC_data{T1<:Int64, T2<:Float64}
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

function compute_new_energy(pol::Magnetic_polymer, near::Array{Int,2}, ff::Array{Float64}, s::Array{Int}, J::Float64)
    ene = 0.0
    ene_J = 0.0
    for i_mono in 1:pol.n_mono
        ene -= ff[i_mono] * s[i_mono]
        for j in 1:near[1, i_mono]
            ene_J -= s[i_mono] * s[near[j+1,i_mono]] * J * 0.5
        end
    end
    return ene + ene_J, ene_J
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
function spins_MC!(pol::Magnetic_polymer, n_flips::Int, ff::Array{Float64}, s::Array{Int}, near::Array{Int,2}, J::Float64, inv_temp::Float64)
    delta_tot = 0.0
    delta_tot_J = 0.0
    for i_flip in 1:n_flips
        flip_candidate = rand(1:pol.n_mono)
        local_ene_J = 0.0
        local_ene = -ff[flip_candidate] * s[flip_candidate]
        for j in 1:near[1,flip_candidate]
            local_ene_J -= J * s[flip_candidate] * s[near[j+1,flip_candidate]]
        end
        local_ene += local_ene_J

        #s[flip_candidate]==1 ? trial_spin_value=-1 : trial_spin_value=1
        s[flip_candidate]==1 ? trial_spin_value=0 : trial_spin_value=1

        trial_local_ene_J = 0.0
        trial_local_ene = -ff[flip_candidate] * trial_spin_value
        for j in 1:near[1,flip_candidate]
            trial_local_ene_J -= J * trial_spin_value * s[near[j+1,flip_candidate]]
        end
        trial_local_ene += trial_local_ene_J

        delta_ene = trial_local_ene - local_ene
        acc = 0.0

        delta_ene<=0 ? acc=1.0 : acc=exp(-delta_ene*inv_temp)

        if acc > rand()
            delta_tot += delta_ene
            delta_tot_J += (trial_local_ene_J - local_ene_J)
            s[flip_candidate] = trial_spin_value
        end
    end
    return delta_tot, delta_tot_J
end


#########################################################################
#########################################################################

#= This is the version of MC_run where I don't save the configurations of the spins (only the last one)
   This is useful if I just want to perform the simulation and I am interested only in storing the 
    values of some observables along the markov chain. If instead I want to carry out some simulator 
    based inference task, I will also need to save spin configurations. Expoiting julia's 
    multiple dispatch feature I can always use the same "MC_run!()" function, the specific method 
    is inferred by the compiler depending on the list of argument types. "MC_run_save!()" is the function
    that save the configurations and it will also be made into a MC_run!() module=#

function MC_run_base!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int, inv_temp::Float64, 
                J::Float64, fields::Array{Float64}, quenched_spins::Bool=false)
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
    energy, _ = compute_new_energy(pol,pol.neighbours,fields, pol.spins, J)

    traj.energies[start] = energy
    traj.magnetization[start] = mean(pol.spins)
    traj.rg2[start] = gyration_radius_squared(pol)

    empty!(pol.hash_saw)
    
    for i_step in (start+1):finish
        pivot = rand(2:pol.n_mono-1)
        p_move = rand(1:47)
        if i_step%50000 == 0
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
        trial_energy, _ = compute_new_energy(pol,pol.trial_neighbours,fields, pol.spins, J)
        delta_energy = trial_energy - energy
        acc = 0.0
        delta_energy<=0 ? acc=1.0 : acc=exp(-delta_energy*inv_temp)
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
        
        if !quenched_spins
            delta_ene, _ = spins_MC!(pol, pol.n_mono,fields,pol.spins,pol.neighbours, J, inv_temp) ## spins_MC! return the total energy variation of the accepted spin n_flips
            energy += delta_ene
        end

        traj.rg2[i_step] = gyration_radius_squared(pol)
        traj.magnetization[i_step] = mean(pol.spins)
        traj.energies[i_step] = energy

        # This next if ... end can be removed. I kept as a sanity check in an early stage 
        # when I had problems with the computation of the nearest neighbours
        #=if !isinteger(energy)
            println("Non integer energy:  ", energy)
            println(pol.coord)
            println(pol.neighbours .- pol.trial_neighbours)
            println(pol.trial_neighbours)
            break
        end=#

        empty!(pol.hash_saw)
    end
end


#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################

# This function is like MC_run_base! but also save configurations
function MC_run_save!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int, spins_configs::Matrix{Int}, 
                sample_lag::Int, n_samples::Int, inv_temp::Float64, J::Float64, fields::Array{Float64}, quenched_spins::Bool=false)
    for i_mono in 1:pol.n_mono
        for j in 1:3
            pol.trial_coord[j,i_mono] = pol.coord[j,i_mono]
        end
    end 
    new_coord1 = zeros(Int,3)
    new_coord2 = zeros(Int,3)
    n_acc = 0
    n_pivots = 0

    hash_base = 2*pol.n_mono + 1

    checksaw!(1, pol, hash_base)
    compute_neighbours!(pol.coord, pol.neighbours, pol.hash_saw, hash_base, pol.n_mono)
    energy,_ = compute_new_energy(pol,pol.neighbours,fields, pol.spins, J)

    traj.energies[start] = energy
    traj.magnetization[start] = mean(pol.spins)
    traj.rg2[start] = gyration_radius_squared(pol)

    empty!(pol.hash_saw)

    if start%sample_lag == 0
        i_sample = div(start,sample_lag)
        for i_mono in 1:pol.n_mono
            spins_configs[i_mono, i_sample]  = pol.spins[i_mono]
        end
    end
    
    for i_step in (start+1):min(finish,sample_lag*n_samples)
        pivot = rand(2:pol.n_mono-1)
        p_move = rand(1:47)
        if i_step%50000 == 0
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
        trial_energy,_ = compute_new_energy(pol,pol.trial_neighbours,fields, pol.spins, J)
        delta_energy = trial_energy - energy
        acc = 0.0
        delta_energy<=0 ? acc=1.0 : acc=exp(-delta_energy*inv_temp)
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
        
        if !quenched_spins
            delta_ene, _ = spins_MC!(pol, pol.n_mono,fields,pol.spins,pol.neighbours, J, inv_temp) ## spins_MC! return the total energy variation of the accepted spin n_flips
            energy += delta_ene
        end

        traj.rg2[i_step] = gyration_radius_squared(pol)
        traj.magnetization[i_step] = mean(pol.spins)
        traj.energies[i_step] = energy

        #=if !isinteger(energy)
            println("Non integer energy:  ", energy)
            println(pol.coord)
            println(pol.neighbours .- pol.trial_neighbours)
            println(pol.trial_neighbours)
            break
        end=#
        empty!(pol.hash_saw)
        if i_step%sample_lag == 0
            i_sample = div(i_step,sample_lag)
            for i_mono in 1:pol.n_mono
                spins_configs[i_mono, i_sample]  = pol.spins[i_mono]
            end
        end
    end
end



#########################################################################
#########################################################################

function MC_run_save!(pol::Magnetic_polymer, traj::MC_data, start::Int, finish::Int, spins_configs::Matrix{Int},
                ising_energies::Array{Float64}, sample_lag::Int, n_samples::Int, inv_temp::Float64, J::Float64, fields::Array{Float64}, quenched_spins::Bool=false)
    for i_mono in 1:pol.n_mono
        for j in 1:3
            pol.trial_coord[j,i_mono] = pol.coord[j,i_mono]
        end
    end 
    new_coord1 = zeros(Int,3)
    new_coord2 = zeros(Int,3)
    n_acc = 0
    n_pivots = 0

    hash_base = 2*pol.n_mono + 1

    checksaw!(1, pol, hash_base)
    compute_neighbours!(pol.coord, pol.neighbours, pol.hash_saw, hash_base, pol.n_mono)
    energy, ising_energy = compute_new_energy(pol,pol.neighbours,fields, pol.spins, J)

    traj.energies[start] = energy
    traj.magnetization[start] = mean(pol.spins)
    traj.rg2[start] = gyration_radius_squared(pol)

    empty!(pol.hash_saw)

    if start%sample_lag == 0
        i_sample = div(start,sample_lag)
        for i_mono in 1:pol.n_mono
            spins_configs[i_mono, i_sample]  = pol.spins[i_mono]
        end
        ising_energies[i_sample] = -ising_energy/J   # actually I save the negative
    end
    
    for i_step in (start+1):min(finish,sample_lag*n_samples)
        pivot = rand(2:pol.n_mono-1)
        p_move = rand(1:47)
        if i_step%50000 == 0
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
        trial_energy, trial_energy_J = compute_new_energy(pol,pol.trial_neighbours,fields, pol.spins, J)
        delta_energy = trial_energy - energy
        acc = 0.0
        delta_energy<=0 ? acc=1.0 : acc=exp(-delta_energy*inv_temp)
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
            ising_energy = trial_energy_J
            n_acc += 1
        end
        
        if !quenched_spins
            delta_ene, delta_ene_J = spins_MC!(pol, pol.n_mono,fields,pol.spins,pol.neighbours, J, inv_temp) ## spins_MC! return the total energy variation of the accepted spin n_flips
            energy += delta_ene
            ising_energy += delta_ene_J
        end
        traj.rg2[i_step] = gyration_radius_squared(pol)
        traj.magnetization[i_step] = mean(pol.spins)
        traj.energies[i_step] = energy

        #=if !isinteger(energy)
            println("Non integer energy:  ", energy)
            println(pol.coord)
            println(pol.neighbours .- pol.trial_neighbours)
            println(pol.trial_neighbours)
            break
        end=#
        empty!(pol.hash_saw)
        if i_step%sample_lag == 0
            i_sample = div(i_step,sample_lag)
            for i_mono in 1:pol.n_mono
                spins_configs[i_mono, i_sample]  = pol.spins[i_mono]
            end
            ising_energies[i_sample] = -ising_energy/J
        end
    end
end


#########################################################################
#########################################################################
#########################################################################
# Multiple Markov Chains

function MMC_run_base!(polymers::Array{Magnetic_polymer}, trajs::Array{MC_data},
                    n_strides::Int, stride::Int, inv_temps::Array{Float64}, J::Float64, fields::Array{Float64}, quenched_spins::Bool=false)
    n_temps = length(inv_temps)
    accepted_swaps = 0
    temp_coord = zeros(Int,3)

    for i_temp in 1:n_temps
        MC_run!(polymers[i_temp], trajs[i_temp],1,stride, inv_temps[i_temp], J, fields, quenched_spins)
    end

    for i_strides in 2:n_strides
        swap = rand(1:n_temps-1)
        delta_ene = (inv_temps[swap] - inv_temps[swap+1]) * (trajs[swap+1].energies[(i_strides-1)*stride] - trajs[swap].energies[(i_strides-1)*stride])
        alpha_swap = 0.0
        delta_ene<=0 ? alpha_swap=1.0 : alpha_swap=exp(-delta_ene)
        if alpha_swap >= rand()
            for i_mono in 1:polymers[1].n_mono
                for j in 1:3
                   temp_coord[j] = polymers[swap].coord[j,i_mono] 
                   polymers[swap].coord[j,i_mono] = polymers[swap+1].coord[j,i_mono]
                   polymers[swap+1].coord[j,i_mono] =  temp_coord[j]
                end
                temp_spin = polymers[swap].spins[i_mono]
                polymers[swap].spins[i_mono] = polymers[swap+1].spins[i_mono]
                polymers[swap+1].spins[i_mono] = temp_spin
            end
            accepted_swaps += 1
        end

        for i_temp in 1:n_temps
            MC_run!(polymers[i_temp], trajs[i_temp],(i_strides-1)*stride+1,i_strides*stride, inv_temps[i_temp], J, fields, quenched_spins)
        end
    end
    println("Accepted_swaps: ", accepted_swaps)
end

function MMC_run_save!(polymers::Array{Magnetic_polymer}, trajs::Array{MC_data},
                    n_strides::Int, stride::Int, inv_temps::Array{Float64}, spins_configs::Matrix{Int}, sample_lag::Int, n_samples::Int, J::Float64, fields::Array{Float64}, quenched_spins::Bool=false)
    n_temps = length(inv_temps)
    accepted_swaps = 0
    temp_coord = zeros(Int,3)

    MC_run!(polymers[1], trajs[1],1,stride, spins_configs,sample_lag, n_samples, inv_temps[1], J, fields, quenched_spins)
    # Only the lowest temperature is the system we're using for our likelihood approx
    # the chains at higher temps are only used to "fluidify" the chain of interest
    for i_temp in 2:n_temps
        MC_run!(polymers[i_temp], trajs[i_temp],1,stride, sample_lag, n_samples, inv_temps[i_temp], J, fields, quenched_spins)
    end

    for i_strides in 2:n_strides
        swap = rand(1:n_temps-1)
        delta_ene = (inv_temps[swap] - inv_temps[swap+1]) * (trajs[swap+1].energies[i_strides*stride] - trajs[swap].energies[i_strides*stride])
        alpha_swap = 0.0
        delta_ene<=0 ? alpha_swap=1.0 : alpha_swap=exp(-delta_ene)
        if alpha_swap >= rand()
            for i_mono in 1:polymers[1].n_mono
                for j in 1:3
                   temp_coord[j] = polymers[swap].coord[j,i_mono] 
                   polymers[swap].coord[j,i_mono] = polymers[swap+1].coord[j,i_mono]
                   polymers[swap+1].coord[j,i_mono] =  temp_coord[j]
                end
                
            end
            if !quenched_spins
                for i_mono in 1:polymers[1].n_mono
                    temp_spin = polymers[swap].spins[i_mono]
                    polymers[swap].spins[i_mono] = polymers[swap+1].spins[i_mono]
                    polymers[swap+1].spins[i_mono] = temp_spin
                end
            end
            accepted_swaps += 1
        end
        MC_run!(polymers[1], trajs[1],(i_strides-1)*stride+1,i_strides*stride, spins_configs,sample_lag, n_samples, inv_temps[1], J, fields, quenched_spins)
        # Only the lowest temperature is the system we're using for our likelihood approx
        # the chains at higher temps are only used to "fluidify" the chain of interest
        for i_temp in 2:n_temps
            MC_run!(polymers[i_temp], trajs[i_temp],(i_strides-1)*stride+1,i_strides*stride, sample_lag, n_samples, inv_temps[i_temp], J, fields, quenched_spins)
        end
    end
    println("Accepted_swaps: ", accepted_swaps)
end

### This one calls for the lower temperature chain the MC_run() method that also saves the ising energies
function MMC_run_save!(polymers::Array{Magnetic_polymer}, trajs::Array{MC_data},
                    n_strides::Int, stride::Int, inv_temps::Array{Float64}, spins_configs::Matrix{Int}, ising_energies::Array{Float64}, sample_lag::Int, n_samples::Int, J::Float64, fields::Array{Float64}, quenched_spins::Bool=false)
    n_temps = length(inv_temps)
    accepted_swaps = 0
    temp_coord = zeros(Int,3)

    MC_run!(polymers[1], trajs[1],1,stride, spins_configs, ising_energies, sample_lag, n_samples, inv_temps[1], J, fields, quenched_spins)
    # Only the lowest temperature is the system we're using for our likelihood approx
    # the chains at higher temps are only used to "fluidify" the chain of interest
    for i_temp in 2:n_temps
        MC_run!(polymers[i_temp], trajs[i_temp],1,stride, sample_lag, n_samples, inv_temps[i_temp], J, fields, quenched_spins)
    end

    for i_strides in 2:n_strides
        swap = rand(1:n_temps-1)
        delta_ene = (inv_temps[swap] - inv_temps[swap+1]) * (trajs[swap+1].energies[i_strides*stride] - trajs[swap].energies[i_strides*stride])
        alpha_swap = 0.0
        delta_ene<=0 ? alpha_swap=1.0 : alpha_swap=exp(-delta_ene)
        if alpha_swap >= rand()
            for i_mono in 1:polymers[1].n_mono
                for j in 1:3
                    temp_coord[j] = polymers[swap].coord[j,i_mono] 
                    polymers[swap].coord[j,i_mono] = polymers[swap+1].coord[j,i_mono]
                    polymers[swap+1].coord[j,i_mono] =  temp_coord[j]
                end
            end
            if !quenched_spins
                for i_mono in 1:polymers[1].n_mono
                    temp_spin = polymers[swap].spins[i_mono]
                    polymers[swap].spins[i_mono] = polymers[swap+1].spins[i_mono]
                    polymers[swap+1].spins[i_mono] = temp_spin
                end
            end
            accepted_swaps += 1
        end
        MC_run!(polymers[1], trajs[1],(i_strides-1)*stride+1,i_strides*stride, spins_configs, ising_energies, sample_lag, n_samples, inv_temps[1], J, fields, quenched_spins)
        # Only the lowest temperature is the system we're using for our likelihood approx
        # the chains at higher temps are only used to "fluidify" the chain of interest
        for i_temp in 2:n_temps
            MC_run!(polymers[i_temp], trajs[i_temp],(i_strides-1)*stride+1,i_strides*stride, sample_lag, n_samples, inv_temps[i_temp], J, fields, quenched_spins)
        end
    end
    println("Accepted_swaps: ", accepted_swaps)
end
