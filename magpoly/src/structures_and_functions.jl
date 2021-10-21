using Random, Distributions
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
            
            #println("i,j=",i,", ",j)
            #println("p_moves_count= ", p_moves_count)
            #println(pivot_moves[p_moves_count,:,:])
            global p_moves_count += 1
        end
    end
end
#for i in 1:47
#    println(pivot_moves[i,:,:])
#end
#########################################################################
#########################################################################

# This structure contains all I need to compute the Hamiltonian
struct Magnetic_polymer{T1<:Int, T2<:AbstractFloat}
    n_mono::T1
    inv_temp::T2
    J::T2
    alpha::T2
    spins::Array{T1}
    fields::Array{T2}
    coord::Array{T1,2}         # Coordinates of the monomers
    trial_coord::Array{T1,2}
    hash_saw::Dict{Int64, Int64}
    neighbours::Array{T1, 2}
    trial_neighbours::Array{T1, 2}
end

function Magnetic_polymer(n_mono::T1, inv_temp::T2, J::T2, alpha::T2) where {T1<:Int, T2<:AbstractFloat}
    Magnetic_polymer(n_mono, inv_temp, J, alpha, zeros(Int, n_mono), zeros(n_mono), zeros(Int, 3,n_mono),
    zeros(Int, 3,n_mono), Dict{Int, Int}(), zeros(Int, 7,n_mono), zeros(Int, 7,n_mono))
end

function initialize_poly!(poly::Magnetic_polymer)
    d = Normal(0,2)
    poly.spins .= rand(0:1, poly.n_mono)
    poly.fields .= rand(d, poly.n_mono)
    for i_mono in 1:poly.n_mono
        poly.coord[1, i_mono] = i_mono
    end    
    for i_mono in 1:poly.n_mono
        for j in 1:3
            poly.trial_coord[j, i_mono] = poly.coord[j, i_mono]
        end
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
    re2::Array{T1}
end

function MC_data(n_steps::Int)
    MC_data(n_steps, zeros(n_steps), zeros(n_steps), zeros(n_steps), zeros(Int, n_steps))
end

#########################################################################
#########################################################################


function try_pivot!(k::Int, move::Int, pol::Magnetic_polymer)
    for i_mono in 1:k
        for j in 1:3
            pol.trial_coord[j, i_mono] = pol.coord[j, i_mono]
        end
    end
    for i_mono in k+1:pol.n_mono
        for j in 1:3
            pol.trial_coord[j,i_mono] = (pivot_moves[move,2,j]*(pol.coord[pivot_moves[move,1,j],i_mono]
                                         - pol.coord[pivot_moves[move,1,j],k]) + pol.coord[j,k])
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
        near[2:7, i_mono] .= -1

        neigh_coord = coo[:, i_mono] # remember that slices in julia create a copy

        for j in 1:3
            for k in -1:2:1
                neigh_coord[j] += k
                saw_key = neigh_coord[1]*a^2 + neigh_coord[2]*a + neigh_coord[3]
                if haskey(dic, saw_key)
                    near[1, i_mono] += 1
                    n_neigh = near[1, i_mono]+1
                    near[n_neigh, i_mono] = dic[saw_key]
                end
                neigh_coord[j] -= k
            end
        end
    end
end

#########################################################################

function gyration_radius_squared(pol::Magnetic_polymer)
    #rcm = sum(pol.coord, dims=2)./pol.n_mono
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
#########################################################################

function MC_run!(pol::Magnetic_polymer, traj::MC_data)
    n_acc = 0
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

    #checksaw!(1, pol, hash_base)
    #compute_neighbours!(pol.coord, pol.neighbours, pol.hash_saw, hash_base, pol.n_mono)
    #println(pol.hash_saw)
    #for i in 1:pol.n_mono
    #    println(view(pol.neighbours,:,i))
    #end

    for i_step in 1:traj.n_steps
        pivot = rand(2:pol.n_mono-1)
        p_move = rand(1:47)
        if i_step%100000 == 0
            println("Pivot: ", pivot)
            println("Attmpted move: ", p_move)
            println("step number: ", i_step, "\n")
        end
        try_pivot!(pivot, p_move, pol)
        still_saw = checksaw!(pivot, pol, hash_base)
        if still_saw
            n_acc += 1
            for i_mono in 1:pol.n_mono
                for j in 1:3
                    pol.coord[j,i_mono] = pol.trial_coord[j,i_mono]
                end
            end
        end
        for j in 1:3
            traj.re2[i_step] += pol.coord[j,end]^2 
        end
        traj.rg2[i_step] = gyration_radius_squared(pol)
        empty!(pol.hash_saw)
        
    end
    println("Fraction of accepted moves: ", n_acc/traj.n_steps)
end



