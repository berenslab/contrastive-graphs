using SparseArrays
using NPZ
using SGtSNEpi
using ArgParse

function (@main)(ARGS)
    s = ArgParseSettings()
    @add_arg_table! s begin
        "--row"
        help = "row.npy file of the sparse matrix"
        required = true
        "--col"
        help = "col.npy file of the sparse matrix"
        required = true
        "--data"
        help = "data.npy file of the sparse matrix"
        required = true
        "--shape"
        help = "shape of the sparse adjacency (square) matrix"
        arg_type = UInt
        default = 0
        "--init"
        help = ".npy file for the initial layout for sgtsnepi"
        required = true
        "-d", "--dim"
        help = "Number of dimensions to pass to sgtsnepi (parameter `d`)"
        arg_type = Int
        default = 2
        dest_name = "d"
        "--max_iter"
        help = "Parameter `max_iter` in sgtsnepi"
        arg_type = Int
        default = 1000
        "--early_exag"
        help = "Number of early exag. iterations"
        default = 250
        arg_type = Int
        "--learning_rate"
        help = "LR parameter `eta` in sgtsnepi"
        default = 200
        arg_type = Float64
        dest_name = "eta"
        "--λ", "-λ", "--lambda"
        help = "Rescaling parameter `λ` in sgtsnepi"
        arg_type = Float64
        default = 10
        "--alpha", "-ɑ", "--ɑ"
        help = "Early exag. multiplier `α` in sgtsnepi"
        arg_type = Float64
        default = 12
        "--np"
        help = "Number of processes to use, use 0 for all available"
        arg_type = Int
        default = 0
        "--outfile"
        help = "filename where the result of `sgtsnepi` will be saved"
        required = true
    end
    args = parse_args(ARGS, s)
    data = npzread(pop!(args, "data"))
    col = npzread(pop!(args, "col"))
    row = npzread(pop!(args, "row"))
    Y0 = npzread(pop!(args, "init"))
    n = pop!(args, "shape", 0)
    A = if n > 0
        sparse(row .+ 1, col .+ 1, data, n, n)
    else
        sparse(row .+ 1, col .+ 1, data)
    end

    outfile = pop!(args, "outfile")
    kwargs = Dict(Symbol(k) => v for (k, v) in args)
    Y = sgtsnepi(A; Y0=Y0, kwargs...)

    npzwrite(outfile, Y)

    return 0
end
