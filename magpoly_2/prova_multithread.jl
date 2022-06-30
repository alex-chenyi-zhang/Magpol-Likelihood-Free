function run()
    n_columns = 12
    n_rows = 100000
    M = zeros(n_rows,n_columns)
    for i in 1:n_columns
        M[n_rows,i] = i
    end
    n_acc = 0
    for i_stride in 1:10000
        Threads.@threads for n in 1:n_columns
            for i in 1:n_rows-1
                M[i,n] += rand() - 1.0
            end
        end
        swap = rand(1:n_columns-1)
        delta = sum(M[1:end-1, swap]) - sum(M[1:end-1, swap+1])
        alpha = 0.0
        delta<=0 ? alpha=1.0 : alpha=exp(-delta)
        if alpha >= rand()
            x = M[n_rows, swap]
            M[n_rows, swap] = M[n_rows, swap+1]
            M[n_rows, swap+1] = x
            n_acc += 1
        end
    end
    println(n_acc)
    println(M[n_rows,:])
end

@time run()