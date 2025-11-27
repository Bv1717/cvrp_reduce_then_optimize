module Frank_Wolfe_regularisation

export compute_argmax_relaxed_regularized_CVRP_FW



using MathOptInterface
const MOI = MathOptInterface
using GLPK 
using FrankWolfe
using DifferentiableFrankWolfe
using LinearAlgebra
using PythonCall


function build_vector(varpos; y_arc_vars=[], f_arc_vars=[], 
                      y_values=[], f_values=[])
    v = zeros(length(varpos))  # initialize vector of right length

    for (arc, var) in y_arc_vars
        v[varpos[var]] = y_values[arc]
    end

    # f_arc_vars is a Dict{Tuple, MOI.VariableIndex}
    for (arc, var) in f_arc_vars
        v[varpos[var]] = 0
    end

    return v
end




function feasible_relaxed_CVRP(
    demands ,
    arcs ,
    nb_vehicles ,
    capacity_vehicles ,
    arc_costs 
)

    MOI = MathOptInterface
    optimizer = GLPK.Optimizer()

    n = length(demands)
    depot_index = findfirst(==(0), demands)
    depot_index === nothing && error("No depot found: demands must contain a single 0 demand.")
    depot_index = depot_index::Int

    # Create variables
    y_arc_vars = Dict{Tuple{Int,Int}, MOI.VariableIndex}()
    f_arc_vars = Dict{Tuple{Int,Int}, MOI.VariableIndex}()

    for (i,j) in arcs
        y_arc_vars[(i,j)] = MOI.add_variable(optimizer)
        MOI.add_constraint(optimizer, y_arc_vars[(i,j)], MOI.GreaterThan(0.0))
    end
    for (i,j) in arcs
        f_arc_vars[(i,j)] = MOI.add_variable(optimizer)
        MOI.add_constraint(optimizer, f_arc_vars[(i,j)], MOI.GreaterThan(0.0))
    end

    for i in 1:n
        if i != depot_index
            terms = MOI.ScalarAffineTerm{Float64}[]
            for j in 1:n
                if haskey(y_arc_vars, (i,j))
                    push!(terms, MOI.ScalarAffineTerm(1.0, y_arc_vars[(i,j)]))
                end
            end
            aff = MOI.ScalarAffineFunction(terms, 0.0)
            MOI.add_constraint(optimizer, aff, MOI.EqualTo(1.0))
        end
    end

    for j in 1:n
        if j != depot_index
            terms = MOI.ScalarAffineTerm{Float64}[]
            for i in 1:n
                if haskey(y_arc_vars, (i,j))
                    push!(terms, MOI.ScalarAffineTerm(1.0, y_arc_vars[(i,j)]))
                end
            end
            aff = MOI.ScalarAffineFunction(terms, 0.0)
            MOI.add_constraint(optimizer, aff, MOI.EqualTo(1.0))
        end
    end

    terms2 = MOI.ScalarAffineTerm{Float64}[]
    for i in 1:n
        if i != depot_index && haskey(y_arc_vars,(depot_index, i))
            push!(terms2, MOI.ScalarAffineTerm(1.0, y_arc_vars[(depot_index, i)]))
        end
    end
    aff2 = MOI.ScalarAffineFunction(terms2, 0.0)
    MOI.add_constraint(optimizer, aff2, MOI.LessThan(Float64(nb_vehicles)))

    for (i,j) in arcs
        terms = MOI.ScalarAffineTerm{Float64}[
            MOI.ScalarAffineTerm(1.0, f_arc_vars[(i,j)]),
            MOI.ScalarAffineTerm(-Float64(capacity_vehicles), y_arc_vars[(i,j)])
        ]
        aff = MOI.ScalarAffineFunction(terms, 0.0)
        MOI.add_constraint(optimizer, aff, MOI.LessThan(0.0))
    end

    for i in 1:n
        if i != depot_index
            terms3 = MOI.ScalarAffineTerm{Float64}[]
            for j in 1:n
                if haskey(f_arc_vars, (i,j))
                    push!(terms3, MOI.ScalarAffineTerm(1.0, f_arc_vars[(i,j)]))
                end
                if haskey(f_arc_vars, (i,j))
                    push!(terms3, MOI.ScalarAffineTerm(-1.0, f_arc_vars[(j,i)]))
                end
            end
            aff3 = MOI.ScalarAffineFunction(terms3, 0.0)
            MOI.add_constraint(optimizer, aff3, MOI.EqualTo(Float64(demands[i])))
        end
    end

    # Objective: minimize sum_{(i,j)} arc_costs[i,j] * y_{ij}
    obj_terms = MOI.ScalarAffineTerm{Float64}[]

    for (i,j) in arcs
        haskey(arc_costs, (i,j)) || error("arc_costs missing key")
        push!(obj_terms, MOI.ScalarAffineTerm(arc_costs[(i,j)], y_arc_vars[(i,j)]))
    end
    obj = MOI.ScalarAffineFunction(obj_terms, 0.0)
    MOI.set(optimizer, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), obj)
    MOI.set(optimizer, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    # Optimize
    MOI.optimize!(optimizer)
    term_status = MOI.get(optimizer, MOI.TerminationStatus())
    term_status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED) || @warn "Solver terminated with $term_status"

    # Extract optimal values into dictionaries
    y_opt = Dict{Tuple{Int,Int},Float64}()
    f_opt = Dict{Tuple{Int,Int},Float64}()
    for (i,j) in arcs
        y_opt[(i,j)] = MOI.get(optimizer, MOI.VariablePrimal(), y_arc_vars[(i,j)])
        f_opt[(i,j)] = MOI.get(optimizer, MOI.VariablePrimal(), f_arc_vars[(i,j)])
    end

    return y_opt, f_opt
end







function build_relaxed_CVRP_polytope(demands, arcs ,
    nb_vehicles , capacity_vehicles )

    optimizer = GLPK.Optimizer()

    n = length(demands)

    depot_index = 1

    for (i, demand) in enumerate(demands)
        if demand == 0
            depot_index = i
            break
        end
    end

    f_arc_vars = Dict{Tuple{Int,Int}, MOI.VariableIndex}()
    y_arc_vars = Dict{Tuple{Int,Int}, MOI.VariableIndex}()
    for (i,j) in arcs
        y_arc_vars[(i,j)] = MOI.add_variable(optimizer)
        MOI.add_constraint(optimizer, y_arc_vars[(i,j)], MOI.GreaterThan(0.0))
    end

    for(i,j) in arcs
        f_arc_vars[(i,j)] = MOI.add_variable(optimizer)
    end


    for i in 1:n
        if i != depot_index
            terms = MathOptInterface.ScalarAffineTerm{Float64}[]
            for j in 1:n
                if (i, j) in arcs
                    push!(terms, MOI.ScalarAffineTerm(1.0, y_arc_vars[(i,j)]))
                end
            end
            aff = MOI.ScalarAffineFunction(terms, 0.0)
            MOI.add_constraint(optimizer, aff, MOI.EqualTo(1.0))
        end
    end

    for j in 1:n
        if j != depot_index
            terms = MathOptInterface.ScalarAffineTerm{Float64}[]
            for i in 1:n
                if (i, j) in arcs
                    push!(terms, MOI.ScalarAffineTerm(1.0, y_arc_vars[(i,j)]))
                end
            end
            aff = MOI.ScalarAffineFunction(terms, 0.0)
            MOI.add_constraint(optimizer, aff, MOI.EqualTo(1.0))
        end
    end

    terms2 = MathOptInterface.ScalarAffineTerm{Float64}[]
    for i in 1:n
        if i != depot_index
            if(depot_index, i) in arcs
                push!(terms2, MOI.ScalarAffineTerm(1.0, y_arc_vars[(depot_index, i)]))
            end
        end
    end
    aff2 = MOI.ScalarAffineFunction(terms2, 0.0)
    MOI.add_constraint(optimizer, aff2, MOI.LessThan(Float64(nb_vehicles)))

    for (idx, (i,j)) in enumerate(arcs)
        MOI.add_constraint(optimizer, f_arc_vars[(i,j)], MOI.GreaterThan(0.0))
        terms = [MOI.ScalarAffineTerm(1.0, f_arc_vars[(i,j)]),
         MOI.ScalarAffineTerm(Float64(-capacity_vehicles ), y_arc_vars[(i,j)])]

        aff = MOI.ScalarAffineFunction(terms, 0.0)
        MOI.add_constraint(optimizer, aff, MOI.LessThan(0.0))
    end

    for i in 1:n
        if i != depot_index
            terms3 = MathOptInterface.ScalarAffineTerm{Float64}[]
            for j in 1:n
                if (i,j) in arcs
                    push!(terms3, MOI.ScalarAffineTerm(1.0, f_arc_vars[(i,j)]))
                end
                if (j, i) in arcs
                    push!(terms3, MOI.ScalarAffineTerm(-1.0, f_arc_vars[(j,i)]))
                end
            end
            aff3 = MOI.ScalarAffineFunction(terms3, 0.0)
            MOI.add_constraint(optimizer, aff3, MOI.EqualTo(Float64(demands[i])))
        end
    end

    return optimizer, y_arc_vars, f_arc_vars          
end


function compute_argmax_relaxed_regularized_CVRP_FW(demands,
    arcs_list , arc_costs ,
    nb_vehicles , capacity_vehicles , lambda , 
    max_iteration ) 

    demands = Int.(demands) 

    arcs_list_jl = [(Int(a[1]), Int(a[2])) for a in arcs_list]

    
    arc_costs_jl = Dict{Tuple{Int,Int}, Float64}()

    for (k,v) in arc_costs  # PythonCall lets you iterate PyDict like this
        i, j = Int(k[1]), Int(k[2])
        arc_costs_jl[(i,j)] = float(v)
    end

    cvrp_optimizer, y_arc_vars, f_arc_vars =
    build_relaxed_CVRP_polytope(demands, arcs_list_jl, nb_vehicles, capacity_vehicles)

    y_initial_values, f_initial_values = feasible_relaxed_CVRP(demands, arcs_list_jl,
    nb_vehicles, capacity_vehicles, arc_costs_jl)


    vars = MOI.get(cvrp_optimizer, MOI.ListOfVariableIndices())
    varpos = Dict(v => k for (k,v) in enumerate(vars))

    θ = build_vector(varpos;
        y_arc_vars = y_arc_vars,
        f_arc_vars = f_arc_vars,
        y_values   = arc_costs_jl,
        f_values   = zeros(length(f_arc_vars)) # or whatever default
    )

    y0 = build_vector(varpos;
        y_arc_vars = y_arc_vars,
        f_arc_vars = f_arc_vars,
        y_values   = y_initial_values,
        f_values   = f_initial_values
    )

    lmo = FrankWolfe.MathOptLMO(cvrp_optimizer)

    Ω(y) = lambda*dot(y, y)
    Ω_grad(y) = lambda*2*y

    f(y, θ) = Ω(y) - dot(θ, y)
    f_grad1(y, θ) = Ω_grad(y) - θ

    dfw = DiffFW(f, f_grad1, lmo)

    weights, stats = dfw.implicit(y0, θ, (; max_iteration = max_iteration))
    solution = sum(weights[i] .* stats.active_set.atoms[i] for i in eachindex(weights))

    y_arc_sol = Dict{Tuple{Int,Int}, Float64}()
    f_arc_sol = Dict{Tuple{Int,Int}, Float64}()
    

    sol_vector =[]
    for i in 1:length(y_arc_vars)
        y_arc_sol[arcs_list_jl[i].-(1,1)] = solution[i]
        push!(sol_vector, solution[i])
    end
    for i in 1:length(y_arc_vars)
        f_arc_sol[arcs_list_jl[i].-(1,1)] = solution[i+12]
        push!(sol_vector, solution[i+12])

    end

    sol_value = f(sol_vector, θ)
    return y_arc_sol, f_arc_sol, sol_value
end
end