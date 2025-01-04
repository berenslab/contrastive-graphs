import Pkg
Pkg.activate(".")
pkgs = ["SGtSNEpi", "SparseArrays", "NPZ", "ArgParse", "PackageCompiler"]
Pkg.add(pkgs)
import PackageCompiler

function (@main)(ARGS)
    PackageCompiler.create_sysimage(
        ["SGtSNEpi", "SparseArrays", "NPZ", "ArgParse"],
        sysimage_path=ARGS[1],
        precompile_execution_file="precompile_sgtsnepi.jl",
        import_into_main=true,
    )
end
