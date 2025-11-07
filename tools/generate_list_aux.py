def generate_variable_indices(N_S):
    """Generates a dictionary mapping variable names to their indices based on the number of species (N_S)."""
    
    # Main variables
    indices = {
        "U": 1 + N_S,
        "V": 2 + N_S,
        "W": 3 + N_S,
        "H": 4 + N_S,
        "T": 5 + N_S,
        "P": 6 + N_S,
        "C": 7 + N_S,
        "MU": 8 + N_S,
        "DUC": 9 + N_S,
        "DIV": 10 + N_S,
        "R": 11 + N_S,
        "Z": 12 + N_S,
        "K_COND": 13 + N_S,
    }
    
    # Start of diffusivities index
    J_D_START = 13 + N_S
    # Start of wdot index
    J_WDOT_START = 14 + 2*N_S
    
    # Additional variables
    indices["LES1"] = 14 + 2 * N_S
    indices["HRR"] = 15 + 3 * N_S
    indices["GAM_PASR"] = 16 + 3 * N_S

    # Mass fractions (1 to N_S)
    mass_fractions = {f"Y{i}": i for i in range(1, N_S + 1)}

    # Diffusivities (J_D_START+1 to J_D_START+N_S)
    diffusivities = {f"D{i}": J_D_START + i for i in range(1, N_S + 1)}

    # Source terms (J_WDOT_START+1 to J_WDOT_START+N_S)
    wdots = {f"WDOT{i}": J_WDOT_START + i for i in range(1, N_S + 1)}


    return indices, mass_fractions, diffusivities, wdots

def select_variables():
    """Asks the user for the number of species and which variables they want to save."""
    N_S = int(input("Insert number of species:\n"))
    indices, mass_fractions, diffusivities, wdots = generate_variable_indices(N_S)

    # Display available variables
    print("\nAvailable general variables:")
    for var, idx in indices.items():
        print(f"{var}: {idx}")

    print("\nAvailable mass fractions:")
    for var, idx in mass_fractions.items():
        print(f"{var}: {idx}")

    print("\nAvailable diffusivities:")
    for var, idx in diffusivities.items():
        print(f"{var}: {idx}")

    print("\nAvailable wdots:")
    for var, idx in wdots.items():
        print(f"{var}: {idx}")

    # Ask if the user wants all/none species mass fractions
    include_species = input("\nInclude all mass fractions (Y/N)? ").strip().upper() == "Y"
    include_diffusivities = input("Include all diffusivities (Y/N)? ").strip().upper() == "Y"
    include_wdots = input("Include all wdots (Y/N)? ").strip().upper() == "Y"

    # User input for general variable selection
    selected_vars = input("\nEnter the names of the variables to save (comma-separated, leave blank for none):\n").split(',')
    selected_vars = [var.strip().upper() for var in selected_vars if var.strip()]  # Normalize input

    # Collect indices for selected general variables
    selected_indices = {indices[var] for var in selected_vars if var in indices}

    # Add all species mass fractions if selected
    if include_species:
        selected_indices.update(mass_fractions.values())

    # Add all diffusivities if selected
    if include_diffusivities:
        selected_indices.update(diffusivities.values())

    # Add all wdots if selected
    if include_wdots:
        selected_indices.update(wdots.values())

    # Sort and print indices as a comma-separated string
    sorted_indices = sorted(selected_indices)
    print("\nSelected variable indices:")
    print(",".join(map(str, sorted_indices)))

    return sorted_indices

# Run the function
selected_indices = select_variables()

