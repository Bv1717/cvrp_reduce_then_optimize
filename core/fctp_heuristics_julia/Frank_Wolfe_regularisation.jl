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
    arc_costs, 
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


    # Check termination status
    term_status = MOI.get(optimizer, MOI.TerminationStatus())
    if term_status in (MOI.OPTIMAL, MOI.LOCALLY_SOLVED)
        # println("Solver finished successfully!")

        # Objective value
        obj_val = MOI.get(optimizer, MOI.ObjectiveValue())
        # println("Objective value: ", obj_val)

        # # Retrieve all variable values
        # for ((i,j), var) in y_arc_vars
        #     val = MOI.get(optimizer, MOI.VariablePrimal(), var)
        #     println("y[$i,$j] = $val")
        # end
    else
        @warn "Solver terminated with $term_status"
    end

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
    nb_vehicles , capacity_vehicles, y_hat_dict )

    optimizer = GLPK.Optimizer()

    n = length(demands)

    depot_index = findfirst(==(0), demands)

    y_arc_vars = Dict{Tuple{Int,Int}, MOI.VariableIndex}()
    f_arc_vars = Dict{Tuple{Int,Int}, MOI.VariableIndex}()
    y_arc_order = Tuple{Int,Int}[]

    for (i,j) in arcs
        y_arc_vars[(i,j)] = MOI.add_variable(optimizer)
        if false
            MOI.add_constraint(optimizer, y_arc_vars[(i,j)], MOI.GreaterThan(float(y_hat_dict[(i,j)])))
        else
            MOI.add_constraint(optimizer, y_arc_vars[(i,j)], MOI.GreaterThan(0.0))
        end
        push!(y_arc_order, (i,j))
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

    return optimizer, y_arc_vars, f_arc_vars, y_arc_order          
end


function build_simple_CVRP_polytope(demands, arcs, nb_vehicles, capacity_vehicles)

    optimizer = GLPK.Optimizer()

    n = length(demands)

    depot_index = findfirst(==(0), demands)  # find depot by demand==0

    # Keep ordered list of arcs for consistency
    y_arc_vars = Dict{Tuple{Int,Int}, MOI.VariableIndex}()
    f_arc_vars = Dict{Tuple{Int,Int}, MOI.VariableIndex}()
    y_arc_order = Tuple{Int,Int}[]

    # Arc selection variables y_{ij} ∈ [0,1]
    for (i,j) in arcs
        y_arc_vars[(i,j)] = MOI.add_variable(optimizer)
        MOI.add_constraint(optimizer, y_arc_vars[(i,j)], MOI.GreaterThan(0.0))
        MOI.add_constraint(optimizer, y_arc_vars[(i,j)], MOI.LessThan(1.0))
        push!(y_arc_order, (i,j))
    end

    # Flow variables f_{ij} ∈ [0, capacity_vehicles]
    for (i,j) in arcs
        f_arc_vars[(i,j)] = MOI.add_variable(optimizer)
        MOI.add_constraint(optimizer, f_arc_vars[(i,j)], MOI.GreaterThan(0.0))
        MOI.add_constraint(optimizer, f_arc_vars[(i,j)], MOI.LessThan(Float64(capacity_vehicles)))
    end

    return optimizer, y_arc_vars, f_arc_vars, y_arc_order
end



function f_dict(y::Dict{Tuple{Int,Int},Float64},
                θ::Dict{Tuple{Int,Int},Float64},
                lambda::Float64)

    # squared norm term
    omega = (lambda/2) * sum(y_val^2 for y_val in values(y))

    # linear term: dot(θ, y)
    linear = sum(θ[arc] * y[arc] for arc in keys(y))

    return omega - linear
end

# function find_convergence_index(values::Vector{Float64}; eps=1e-6, window=10)
#     diffs = [abs(values[i+1] - values[i]) for i in 1:length(values)-1]
#     for i in 1:(length(diffs) - window + 1)
#         if all(diffs[i:i+window-1] .< eps)
#             return i + window   # index of first iteration after convergence
#         end
#     end
#     return nothing  # no convergence found
# end

function compute_argmax_relaxed_regularized_CVRP_FW(demands,
    arcs_list, arc_costs,
    nb_vehicles, capacity_vehicles, true_solution, lambda, 
    max_iteration_FW) 

    demands = Int.(demands) 

    arcs_list_jl = [(Int(a[1]), Int(a[2])) for a in arcs_list]

    
    arc_costs_jl = Dict{Tuple{Int,Int}, Float64}()
    true_solution_dict_jl = Dict{Tuple{Int,Int}, Float64}()

    for (k,v) in arc_costs  
        i, j = Int(k[1]), Int(k[2])
        arc_costs_jl[(i,j)] = float(v)
    end

    for (k,v) in true_solution  
        i, j = Int(k[1]), Int(k[2])
        true_solution_dict_jl[(i,j)] = float(v)
    end

    # cvrp_optimizer, y_arc_vars, f_arc_vars, y_arc_order =
    # build_relaxed_CVRP_polytope(demands, arcs_list_jl, nb_vehicles, capacity_vehicles)

    # cvrp_optimizer, y_arc_vars, f_arc_vars, y_arc_order =
    # build_relaxed_CVRP_polytope(demands, arcs_list_jl, nb_vehicles, capacity_vehicles, true_solution_dict_jl)

    # y_initial_values, f_initial_values = feasible_relaxed_CVRP(demands, arcs_list_jl,
    # nb_vehicles, capacity_vehicles, arc_costs_jl, true_solution_dict_jl)


    # vars = MOI.get(cvrp_optimizer, MOI.ListOfVariableIndices())
    # varpos = Dict(v => k for (k,v) in enumerate(vars))

    # θ = build_vector(varpos;
    #     y_arc_vars = y_arc_vars,
    #     f_arc_vars = f_arc_vars,
    #     y_values   = arc_costs_jl,
    #     f_values   = zeros(length(f_arc_vars)) # or whatever default
    # )

    # y0 = build_vector(varpos;
    #     y_arc_vars = y_arc_vars,
    #     f_arc_vars = f_arc_vars,
    #     y_values   = true_solution_dict_jl,
    #     f_values   = f_initial_values
    # )

    # lmo = FrankWolfe.MathOptLMO(cvrp_optimizer)


    # n_arc = length(y_arc_vars)
    # n_flow = length(f_arc_vars)

    # Ω(y) = lambda/2 * dot(y[1:n_arc], y[1:n_arc])

    # function Ω_grad(y)
    #     grad = zeros(length(y))
    #     grad[1:n_arc] .= lambda .* y[1:n_arc]
    #     return grad
    # end

    # f(y, θ) = Ω(y) - dot(θ, y)
    # f_grad1(y, θ) = Ω_grad(y) - θ


    # dfw = DiffFW(f, f_grad1, lmo)

    # weights, stats = dfw.implicit(y0, θ, (; max_iteration = max_iteration))
    # solution = sum(weights[i] .* stats.active_set.atoms[i] for i in eachindex(weights))

    # println("Number of active atoms: ", length(stats.active_set.atoms))
    # for (i, atom) in enumerate(stats.active_set.atoms)
    #     println("Atom $i: norm = ", norm(atom))
    #     println("Weight: ", weights[i])
    # end

    # grad0 = f_grad1(y0, θ)
    # println("Gradient norm at y0: ", norm(grad0))


    # y_arc_sol = Dict{Tuple{Int,Int}, Float64}()
    # f_arc_sol = Dict{Tuple{Int,Int}, Float64}()
    

    # sol_vector =[]
    # for i in 1:length(y_arc_vars)
    #     y_arc_sol[arcs_list_jl[i].-(1,1)] = solution[i]
    #     push!(sol_vector, solution[i])
    # end

    # for i in 1:length(y_arc_vars)
    #     f_arc_sol[arcs_list_jl[i].-(1,1)] = solution[i+12]
    #     push!(sol_vector, solution[i+12])

    # end

    # println("squared_norm : ", Ω(sol_vector))

    # sol_value = f(sol_vector, θ)

    # println("final_value : ", sol_value)

    gradient_dict = Dict{Tuple{Int,Int}, Float64}()

    # println("true_dict : ", true_solution_dict_jl)
    # println("arc_costs_jl :", arc_costs_jl)
    
    # println("true original_value : ", f_dict(true_solution_dict_jl, arc_costs_jl, float(lambda)))

    final_solution_dict_jl = Dict{Tuple{Int,Int}, Float64}()

    zero_cost_dict_jl = Dict{Tuple{Int,Int}, Float64}()

    for (arc, y_val) in arc_costs_jl
        zero_cost_dict_jl[arc] = 0
    end

    y_ini, f_opt_test = feasible_relaxed_CVRP(demands, arcs_list_jl,
        nb_vehicles, capacity_vehicles, zero_cost_dict_jl)

    for (arc, y_val) in y_ini
            final_solution_dict_jl[arc] = y_val                  
    end



    best_score = f_dict(final_solution_dict_jl, arc_costs_jl, float(lambda))
    for t in 1:max_iteration_FW

        for (arc, y_val) in final_solution_dict_jl
            cost = arc_costs_jl[arc]                      
            gradient_dict[arc] = lambda * y_val - cost
        end
        y_opt_test, f_opt_test = feasible_relaxed_CVRP(demands, arcs_list_jl,
        nb_vehicles, capacity_vehicles, gradient_dict)


        for(arc, y_val) in final_solution_dict_jl
            final_solution_dict_jl[arc] = y_val + 2/(t + 3)*(y_opt_test[arc] - y_val)
        end
        test_score = f_dict(final_solution_dict_jl, arc_costs_jl, float(lambda))

        println("test_score : ", test_score)

        if test_score*(1 + 0.001)>best_score
            println("number of iterations : ", t)
            break
        else
            best_score = test_score
        end
    end

    # idx = find_convergence_index(values; eps=1e-6, window=5)
    # println("Converged at iteration index: ", idx)

    # println("final value end: ", f_dict(final_solution_dict_jl, arc_costs_jl, float(lambda)))

    # println("test_norm : ",  (lambda/2) * sum(y_val^2 for y_val in values(final_solution_dict_jl)))

    # println("final_dict : ",final_solution_dict_jl)


    return_solution_dict_jl = Dict{Tuple{Int,Int}, Float64}()
    for (arc, y_val) in final_solution_dict_jl
            return_solution_dict_jl[arc.-(1,1)] = y_val                  
    end

    # print("return_solution_dict_jl : ", return_solution_dict_jl)

    return return_solution_dict_jl
end
end


