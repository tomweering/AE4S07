import sys
import os
import toml
import math
import copy
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata
from ASME._Appendix13_7_a import Appendix13_7_aParams, Appendix13_7_aCalcs
from ASME._Appendix13_8_e import _Appendix13_8_eCalcs
from ASME._Appendix13_7_c import Appendix13_7_cParams, Appendix13_7_cCalcs
from curve import optimize_stiffener_curve


def read_material_data(filename):
    try:
        with open(filename, "r") as file:
            return toml.load(file)
    except FileNotFoundError:
        print("TOML file not found.")
        return {}


def calculate_thermal_equilibrium(material_properties):
    SOLAR_CONSTANT = 1361  # W/m^2
    BOLTZMANN_CONSTANT = 5.670374419e-8  # W/m^2/K^4
    AREA = 0.1 * 0.1  # m^2 for a 10 cm x 10 cm plate

    absorptivity = material_properties["absorptivity"]
    emissivity = material_properties["emissivity"]

    left_side = absorptivity * AREA * SOLAR_CONSTANT
    right_side_factor = emissivity * BOLTZMANN_CONSTANT * AREA

    temperature_kelvin = (left_side / right_side_factor) ** 0.25
    return temperature_kelvin


def interpolate_allowable_stress(material_properties, temperature_celsius):
    room_temp_stress = material_properties.get("allowable_stress_mpa_room_temp", 0)
    elevated_temp_stress = material_properties.get("allowable_stress_mpa_300C", 0)

    if temperature_celsius <= 20:
        return room_temp_stress
    elif temperature_celsius >= 300:
        return elevated_temp_stress

    interpolated_stress = room_temp_stress + (temperature_celsius - 20) * (
        (elevated_temp_stress - room_temp_stress) / (300 - 20)
    )
    return interpolated_stress


def run_pressure_vessel_analysis(material_properties, allowable_stress):
    thickness_range = np.linspace(0.1, 7.5, 500) / 1000  # Thickness values in meters
    pressure_range = np.linspace(0.01, 2, 100) * 10**6  # Pressure values in Pa

    points = []
    labels = []

    for thickness in thickness_range:
        for pressure in pressure_range:
            params_inner = Appendix13_7_aParams(
                long_side_length_inside=(70 / 1000),
                short_side_length_inside=(70 / 1000),
                internal_pressure=pressure,
                short_side_thickness=thickness,
                long_side_thickness=thickness,
                allowable_stress=allowable_stress,
                joint_efficiency=1,
            )

            params_outer = copy.deepcopy(params_inner)
            params_outer.evalAtOuterWalls = True

            calc_inner = Appendix13_7_aCalcs(params_inner)
            calc_outer = Appendix13_7_aCalcs(params_outer)

            max_inner_stress = max(
                abs(calc_inner.S_T_N()), abs(calc_inner.S_T_Q_short()), abs(calc_inner.S_T_M()), abs(calc_inner.S_T_Q_long())
            )
            max_outer_stress = max(
                abs(calc_outer.S_T_N()), abs(calc_outer.S_T_Q_short()), abs(calc_outer.S_T_M()), abs(calc_outer.S_T_Q_long())
            )

            stress_limit = 1.5 * allowable_stress

            is_safe = max_inner_stress <= stress_limit and max_outer_stress <= stress_limit

            points.append((thickness * 1000, pressure / 10**6))  # Convert thickness to mm, pressure to MPa
            labels.append(is_safe)

    return points, labels

def run_rounded_structure_analysis(material_properties, allowable_stress):
    thickness_range = np.linspace(0.1, 7.5, 500) / 1000  # Thickness values in meters
    pressure_range = np.linspace(0.01, 2, 100) * 10**6  # Pressure values in Pa

    points = []
    labels = []

    for thickness in thickness_range:
        for pressure in pressure_range:
            # Define input parameters for Appendix 13-7(c) analysis
            params_inner = Appendix13_7_cParams(
                internal_pressure=pressure,
                corner_radius= 100 / 1000,  # 15 mm in meters
                short_side_half_length=70 / 2 / 1000,  # 35 mm as half length in meters
                long_side_half_length=70 / 2 / 1000,  # 35 mm as half length in meters
                thickness=thickness
            )

            calc_inner = Appendix13_7_cCalcs(params_inner)

            # Evaluate at outer walls
            params_outer = copy.deepcopy(params_inner)
            params_outer.eval_at_outer_walls = True
            calc_outer = Appendix13_7_cCalcs(params_outer)

            # Maximum stress evaluations
            max_inner_stress = max(
                abs(calc_inner.S_T_C()), abs(calc_inner.S_T_D()), abs(calc_inner.S_T_A()), abs(calc_inner.S_T_B())
            )
            max_outer_stress = max(
                abs(calc_outer.S_T_C()), abs(calc_outer.S_T_D()), abs(calc_outer.S_T_A()), abs(calc_outer.S_T_B())
            )

            stress_limit = 1.5 * allowable_stress

            is_safe = max_inner_stress <= stress_limit and max_outer_stress <= stress_limit

            points.append((thickness * 1000, pressure / 10**6))  # Convert thickness to mm, pressure to MPa
            labels.append(is_safe)

    return points, labels


def run_stiffened_structure_analysis(material_properties, allowable_stress, n_stiffeners):
    thickness_range = np.linspace(0.1, 7.5, 500) / 1000  # Thickness values in meters
    pressure_range = np.linspace(0.01, 2, 100) * 10**6  # Pressure values in Pa

    points = []
    labels = []

    side_length = 70 / 1000
    stiffner_thickness = 5 / 1000
    stiffner_height = 5 / 1000
    total_height = 90 / 1000
    #n_stiffeners = 5

    pitch = (total_height - stiffner_height) / (n_stiffeners - 1)

    for thickness in thickness_range:
        for pressure in pressure_range:
            params = {
                "P": pressure,
                "H": 0.07,
                "h": 0.07,
                "t_1": thickness,
                "t_2": thickness,
                "ts_1": stiffner_thickness,
                "ts_2": stiffner_thickness,
                "A_1": 0.07 * stiffner_thickness,
                "A_2": 0.07 * stiffner_thickness,
                "H_1": stiffner_height,
                "h_1": stiffner_height,
                "p": pitch,
                "S": allowable_stress,
                "S_y": material_properties.get("yield_strength_mpa", 0) *10**6,
                "E_2": material_properties.get("modulus_of_elasticity_gpa", 0) *10**9,
                "E_3": material_properties.get("modulus_of_elasticity_gpa", 0) *10**9,
            }

            params_obj = type("Params", (object,), params)()
            calc = _Appendix13_8_eCalcs(params_obj)

            max_stress = max(
                abs(calc.S_T_N()), abs(calc.S_T_Q_short()), abs(calc.S_T_M()), abs(calc.S_T_Q_long())
            )

            stress_limit = 1.5 * allowable_stress

            is_safe = max_stress <= stress_limit

            points.append((thickness * 1000, pressure / 10**6))  # Convert thickness to mm, pressure to MPa
            labels.append(is_safe)

    return points, labels



def plot_pressure_thickness_analysis(materials_data):
    # Ask the user which structure type to plot
    structure_type = input("Enter 'unstiffened' to plot the unstiffened structure, 'stiffened' to plot the stiffened structure, or 'rounded' to plot the rounded structure: ").strip().lower()

    # Create the figures directory if it doesn't exist
    figures_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    plt.subplots_adjust(hspace=0.4, wspace=0.4)

    for ax, (material_name, material_properties) in zip(axes.ravel(), materials_data.items()):
        thermal_equilibrium_temp = calculate_thermal_equilibrium(material_properties)
        allowable_stress = interpolate_allowable_stress(material_properties, thermal_equilibrium_temp)
        allowable_stress = allowable_stress * 10**6  # Convert to Pa

        if structure_type == 'stiffened':
            points, labels = run_stiffened_structure_analysis(material_properties, allowable_stress, n_stiffeners=5)
        elif structure_type == 'rounded':
            points, labels = run_rounded_structure_analysis(material_properties, allowable_stress)
        else:
            points, labels = run_pressure_vessel_analysis(material_properties, allowable_stress)

        points = np.array(points)
        labels = np.array(labels)

        # Prepare grid for contour plot
        thickness = points[:, 0]
        pressure = points[:, 1]
        safe_zone = labels.astype(int)

        grid_thickness, grid_pressure = np.meshgrid(
            np.linspace(thickness.min(), thickness.max(), 200),
            np.linspace(pressure.min(), pressure.max(), 200)
        )

        grid_points = np.column_stack((grid_thickness.ravel(), grid_pressure.ravel()))
        interpolated_labels = griddata(points, safe_zone, grid_points, method="linear")

        # Plot hatched regions
        ax.contourf(
            grid_thickness, grid_pressure, interpolated_labels.reshape(grid_thickness.shape),
            levels=[0, 0.5, 1], colors=["red", "green"], alpha=0.3
        )

        ax.set_title(f"Material: {material_name}")
        ax.set_xlabel("Thickness (mm)")
        ax.set_ylabel("Pressure (MPa)")
        ax.set_xlim(thickness.min(), thickness.max())
        ax.set_ylim(pressure.min(), pressure.max())
        ax.grid(True)

    plt.suptitle(f"{structure_type} Pressure Vessel Safety Analysis for Different Materials")
    
    # Save the plot to the figures directory
    plot_path = os.path.join(figures_dir, f"pressure_thickness_analysis_{structure_type}.png")
    plt.savefig(plot_path)
    plt.show()


def extract_feasible_solution(points, labels, target_pressure=2):
    """
    Extract the first feasible solution for a pressure near 2 MPa
    """
    for idx, point in enumerate(points):
        pressure = point[1]  # Pressure in MPa
        if abs(pressure - target_pressure) < 0.1 and labels[idx]:  # Check near 2 MPa and safe design
            return point[0]  # Return thickness (mm)
    return None


def calculate_mass_cost(material_properties, thickness, n_stiffeners):
    """
    Calculate mass and cost of the design based on thickness and number of stiffeners
    """
    side_length = 70 / 1000
    height = 90 / 1000
    stiffener_thickness = 5 / 1000
    stiffener_height = 5 / 1000
    design_volume = (2 * side_length * thickness + (side_length - 2 * thickness) * thickness) * height

    if n_stiffeners > 0:
        stiffener_volume = 4 * n_stiffeners * side_length * stiffener_thickness * stiffener_height
        design_volume += stiffener_volume

    density = material_properties.get("density_kg_m3", 0)
    price_per_kg = material_properties.get("price_per_kg_eur", 0)

    mass = design_volume * density
    cost = mass * price_per_kg

    return mass, cost


def plot_mass_vs_cost(materials_data):
    """
    Plot mass vs. cost for different materials and stiffener configurations
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    markers = {
        "Unstiffened": "o",
        "2 Stiffeners": "s",
        "3 Stiffeners": "^",
        "4 Stiffeners": "v",
        "5 Stiffeners": "D"
    }

    all_points = []
    material_colors = {}
    color_map = cm.get_cmap('tab10', len(materials_data))

    for idx, (material_name, material_properties) in enumerate(materials_data.items()):
        thermal_equilibrium_temp = calculate_thermal_equilibrium(material_properties)
        allowable_stress = interpolate_allowable_stress(material_properties, thermal_equilibrium_temp)
        allowable_stress *= 10 ** 6

        masses = []
        costs = []
        labels = []

        # Unstiffened structure
        points, safe_labels = run_pressure_vessel_analysis(material_properties, allowable_stress)
        points = np.array(points)
        safe_labels = np.array(safe_labels)

        thickness = extract_feasible_solution(points, safe_labels)
        if thickness:
            thickness_m = thickness / 1000
            mass, cost = calculate_mass_cost(material_properties, thickness_m, n_stiffeners=0)
            masses.append(mass)
            costs.append(cost)
            labels.append("Unstiffened")
            all_points.append((mass, cost, material_name, "Unstiffened", thickness))

        # Stiffened structures
        for n_stiffeners in range(2, 6):
            points, safe_labels = run_stiffened_structure_analysis(material_properties, allowable_stress, n_stiffeners)
            points = np.array(points)
            safe_labels = np.array(safe_labels)

            thickness = extract_feasible_solution(points, safe_labels)
            if thickness:
                thickness_m = thickness / 1000
                mass, cost = calculate_mass_cost(material_properties, thickness_m, n_stiffeners=n_stiffeners)
                masses.append(mass)
                costs.append(cost)
                labels.append(f"{n_stiffeners} Stiffeners")
                all_points.append((mass, cost, material_name, f"{n_stiffeners} Stiffeners", thickness))

        # Assign color to the material
        material_colors[material_name] = color_map(idx)

        # Plot mass vs. cost for each material
        for i, label in enumerate(labels):
            ax.scatter(costs[i], masses[i], color=material_colors[material_name], marker=markers[label])
            #ax.annotate(label, (costs[i], masses[i]))

    # Calculate distances from the origin and print the top 10 closest points
    distances = [((mass**2 + cost**2)**0.5, mass, cost, material_name, structure_type, thickness) for mass, cost, material_name, structure_type, thickness in all_points]
    distances.sort()
    print("Top 20 closest points to the origin (0,0):")
    for i in range(20):
        distance, mass, cost, material_name, structure_type, thickness = distances[i]
        print(f"Material: {material_name}, Structure: {structure_type}, Mass: {mass:.2f} kg, Cost: {cost:.2f} €, Thickness: {thickness:.2f} mm")

    ax.set_title("Mass vs Cost for Different Structures and Materials")
    ax.set_xlabel("Cost (€)")
    ax.set_ylabel("Mass (kg)")
    ax.grid(True)

    # Create custom legend
    handles = [plt.Line2D([0], [0], marker=markers[label], color='w', label=label, markerfacecolor='k', markersize=10) for label in markers]
    handles += [plt.Line2D([0], [0], marker='o', color='w', label=material_name, markerfacecolor=color, markersize=10) for material_name, color in material_colors.items()]
    ax.legend(handles=handles, title="Legend")

    # Save and show plot
    plot_path = os.path.join("figures", "mass_cost_tradeoff.png")
    plt.savefig(plot_path)
    plt.show()

    # Save designs to CSV
    csv_path = os.path.join("data", "designs.csv")
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Material", "Structure Type", "Mass (kg)", "Cost (€)", "Thickness (mm)"])
        for mass, cost, material_name, structure_type, thickness in all_points:
            writer.writerow([material_name, structure_type, mass, cost, thickness])



def capillary_length(material_properties, thickness):
    surface_tension = material_properties.get("surface_tension_mN_per_m", 0)
    density = material_properties.get("density_g_per_cm3", 0)
    #convert to SI units
    surface_tension = surface_tension * 10**-3
    density = density * 10**3

    #assumption on acceleration on 3U cubesat
    m_assumed = 5.5 #[kg]
    T_assumed = 1.0 #[N]
    a = T_assumed / m_assumed #[m/s^2]

    #calculate capillary length
    lamda = (surface_tension / (density * a))**(1/2)

    #get fillet radius to resist capillary effect
    L =math.sqrt((35 - thickness)**2 + (35 - thickness)**2)
    R = L - math.sqrt((lamda**2)/4)

    return lamda, R

#TODO make function that compares PMD (Propellant Management Device) designs
# fillet on the edges to resist the capillary effect
# vanes to transfer fluid from the corners to the center
# sponge to absorb the fluid

#for the vane get surface area from frustum (1/4)lateral area (1/2 for total)
#source: https://www.calculatorsoup.com/calculators/geometry-solids/conicalfrustum.php

def vane_sizing(material_properties):
    flow_rate = 30 #[mL / min] (from MiMPS-G e-micro pump)
    #flow_rate = 420 #[mL / min] (exaggerated)
    viscosity = material_properties.get("absolute_viscosity_mPas", 0)
    density = material_properties.get("density_g_per_cm3", 0)
    surface_tension = material_properties.get("surface_tension_mN_per_m", 0)
    delta = (1.5/1000)  #[m] thickness of "sheet metal vane" (in accordance with printable thickness)
    L = (90/1000)   #[m] length of the vane

    #convert to SI units
    flow_rate = flow_rate * 10**-6 / 60         #[m^3 / s]
    viscosity = viscosity * 10**-3              #[Pa s]
    density = density * 10**3                   #[kg / m^3]
    surface_tension = surface_tension * 10**-3  #[N / m]

    #Slant height of a conical frustum:
    #s = √((r1 - r2)^2 + h^2)
    #Lateral surface area of a conical frustum:
    # S = π * (r1 + r2) * s = π * (r1 + r2) * √((r1 - r2)^2 + h^2)
    #Volume of a conical frustum:
    #V = (1/3) * π * h * (r1^2 + r2^2 + (r1 * r2))

    vanes = []

    #Get relation for Rdown and Rup from knowing total height and overhang angle can at most be 45 degrees
    for R in np.arange(delta/2, 10*delta, 0.001):
        for theta in np.arange(0, math.pi/4, 0.001):
            R_down = R
            R_up = R + L * math.tan(theta)

            s = math.sqrt((R_up - R_down)**2 + L**2)
            S = math.pi * (R_up + R_down) * s
            V = (1/3) * math.pi * L * (R_up**2 + R_down**2 + (R_up * R_down))
            A = S / 2                               #[m^2] (1/2 of the TOTAL lateral area)
            #Q = flow_rate / 4                       #assume 4 vanes, one for each corner
            Q = flow_rate


            viscous_losses = (2*viscosity*Q*L)/A    #assume wetted area = surface area
            dynamic_losses = (density*Q**2)/(2*A**2)

            young_laplace = surface_tension * (1/R_down - 1/R_up)

            V_vane = R_up*delta*L + L*(R_up+ delta/2)**2  #assumption on volume of vane

            #Print pressures
            #print(f"viscous_losses: {viscous_losses}, dynamic_losses: {dynamic_losses}, young_laplace: {young_laplace}")

            #Check if absolute value of yound_laplace is greater than combined absolute value of viscous and dynamic losses
            if abs(young_laplace) > abs(viscous_losses + dynamic_losses):
                vane = [R_down, R_up, V_vane]
                vanes.append(vane)


    #Sort vanes based on V_vane and return the smallest one
    min_vane = min(vanes, key=lambda x: x[2])
    #print(f"R_down: {min_vane[0]*1000} mm, R_up: {min_vane[1]*1000} mm")
    #print(f"viscous_losses: {viscous_losses}, dynamic_losses: {dynamic_losses}, young_laplace: {young_laplace}")

    return min_vane

def sponge_sizing(propellant_properties, thickness):
    # sigma, rho, a, V_req, r_sponge, t=0.3e-3, g_min_fab=0.5e-3, g_max_ratio=2
    propellant_name = propellant_properties.get("name", "")

    # Obtain propellant properties
    sigma = propellant_properties.get("surface_tension_mN_per_m", 0) * 10**-3
    rho = propellant_properties.get("density_g_per_cm3", 0) * 10**3
    # assumption on acceleration on 3U cubesat
    m_assumed = 5.5 #[kg]
    T_assumed = 1.0 #[N]
    a = T_assumed / m_assumed #[m/s^2]

    # Get V_req through burntime and flowrate
    burntime = 10 #[s] (estimate)
    flowrate = 30 #[mL / min] (from MiMPS-G e-micro pump)
    V_req = (flowrate * 10**-6 / 60) * burntime #[m^3]

    P_bubble_point = 200   # Bubble point pressure (Pa) (assumption)
    #ratio = 47438           # Get thickness through simple cantilever model s source: https://apps.dtic.mil/sti/pdfs/AD1098357.pdf

    sponges = []

    # Cycle through a range of radii and thicknesses
    for N in range(10,36):
        r = (2* sigma)/P_bubble_point
        g_max = math.sqrt(sigma/(a*rho))
        R = (g_max*N)/(2*math.pi)
        h = V_req/(math.pi*r**2)

        #TODO Add requirement to get thickness by setting constraint on percentage of total volume dedicated to PMD
        V_tank = ((70/1000)-2*thickness)**2 * (90/1000 - 2*thickness)
        V_PMD = 0.1 * V_tank
        t = V_PMD / (N*(R-r))

        if h > 90e-3:
            h = 90e-3
            r = math.sqrt((V_req)/(h*math.pi))
            r_lattice = (2* sigma)/P_bubble_point
            

        if R < 0.05:
            sponge = [propellant_name, r, R, h, t, N, V_PMD, r_lattice]
            sponges.append(sponge)

    # Sort sponges based on V and return the smallest one
    best_design = min(sponges, key=lambda x: x[5])

    if best_design[2] >= 90e-3:
        print("r too large to rely on surface tension or capillary action to prevent gas injection. Lattice is required.")

    return best_design


def PMD_design():
    # Material densities
    densities = {"Ti6Al4V": 4420, "StainlessSteel316L": 8000, "StainlessSteel304L": 7930, "Inconel625": 8440}
    structure_types = {"Unstiffened": 0, "2 Stiffeners": 2, "3 Stiffeners": 3, "4 Stiffeners": 4, "5 Stiffeners": 5}

    designs_file_path = os.path.join("data", "designs.csv")
    output_file_path = os.path.join("data", "PMD_designs.csv")

    infill = 0.15  # assumed for support and non-structural 15% infill

    with open(designs_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        designs = list(reader)

    with open(output_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Material", "Structure Type", "Propellant", "Thickness (mm)", "R_down (mm)", "R_up (mm)", "V_vane (m^3)", "Vane Mass (kg)", "r_sponge (mm)", "R_sponge (mm)", "h_sponge (mm)", "t_sponge (mm)", "N_sponge", "V_sponge (m^3)", "r_lattice (mm)", "Sponge Mass (kg)", "Support Mass (kg)", "Support Cost (€)", "Total Mass (kg)", "Total Cost (€)"])

        for design in designs:
            material = design["Material"]
            cost = float(design["Cost (€)"])
            mass = float(design["Mass (kg)"])
            price_per_kg = cost / mass
            structure_type = design["Structure Type"]
            thickness = float(design["Thickness (mm)"]) / 1000  # Convert thickness to meters
            density_material = densities.get(material, 0)

            propellant_data = read_material_data(os.path.join(base_dir, "data", "propellants.toml"))
            if propellant_data:
                propellants = propellant_data.get("propellants", {})
                for propellant_name, propellant_properties in propellants.items():

                    lamda, R = capillary_length(propellant_properties, thickness)
                    lamda = lamda / 1000  # Convert from mm to m
                    R = R / 1000  # Convert from mm to m

                    h = 90 / 1000  # Height in meters

                    # Calculate the volume of the fillet
                    volume_fillet = 0.5 * R * h

                    # Calculate the mass of the fillet assuming 15% infill
                    mass_fillet = volume_fillet * density_material * infill

                    # Get mass of support for stiffeners
                    if structure_type == "Unstiffened":
                        n_stiffeners = 0
                        mass_support = 0
                    else:
                        n_stiffeners = structure_types.get(structure_type, 0)
                        area_support = optimize_stiffener_curve(L=90, h=5, n=n_stiffeners, t=5, plot=False)[1]
                        volume_support = 4 * (n_stiffeners - 1) * 70 * area_support
                        volume_support = volume_support / 10**9  # Convert from mm^3 to m^3
                        mass_support = volume_support * density_material * infill

                    vane_design = vane_sizing(propellant_properties)
                    vane_mass = vane_design[2] * density_material

                    sponge_design = sponge_sizing(propellant_properties, thickness)
                    sponge_mass = sponge_design[6] * density_material

                    total_mass = vane_mass + sponge_mass + mass
                    support_cost = mass_support * price_per_kg
                    total_cost = total_mass * price_per_kg + support_cost

                    #TODO limit the accuracy to 0.00 i.e. 3 significant figures
                    writer.writerow([material, structure_type, propellant_name, thickness * 1000, vane_design[0] * 1000, vane_design[1] * 1000, vane_design[2], vane_mass, sponge_design[1] * 1000, sponge_design[2] * 1000, sponge_design[3] * 1000, sponge_design[4] * 1000, sponge_design[5], sponge_design[6], sponge_design[7] * 1000, sponge_mass, mass_support, support_cost, total_mass, total_cost])

                    #print(f"Material: {material}, Structure Type: {structure_type}, Propellant: {propellant_name}, Thickness: {thickness * 1000} mm, R_down: {vane_design[0] * 1000} mm, R_up: {vane_design[1] * 1000} mm, V_vane: {vane_design[2]} m^3, Vane Mass: {vane_mass} kg, r_sponge: {sponge_design[1] * 1000} mm, R_sponge: {sponge_design[2] * 1000} mm, h_sponge: {sponge_design[3] * 1000} mm, t_sponge: {sponge_design[4] * 1000} mm, N_sponge: {sponge_design[5]}, V_sponge: {sponge_design[6]} m^3, r_lattice: {sponge_design[7] * 1000} mm, Sponge Mass: {sponge_mass} kg, Support Mass: {mass_support} kg, Support Cost: {support_cost} €, Total Mass: {total_mass} kg, Total Cost: {total_cost} €")

# Normalize using min-max scaling
def normalize(values):
    min_val, max_val = min(values), max(values)
    return [(v - min_val) / (max_val - min_val) if max_val > min_val else 0 for v in values]

def trade_off():
    # Dictionary of volumetric ISPs (gs/cm³)
    volumetric_specific_impulse_gs_per_cm3 = {
        "AF_M315E": 391, "LMP_103S": 312.48, "HNP225": 245, "FLP_106": 344.6
    }

    # Read CSV file
    designs_file_path = os.path.join("data", "PMD_designs.csv")
    with open(designs_file_path, mode='r') as file:
        reader = csv.DictReader(file)
        designs = list(reader)

    # Extract values for normalization
    mass_values = []
    eff_tank_values = []
    cost_values = []
    r_lattice_values = []
    t_sponge_values = []
    support_mass_values = []

    for design in designs:
        mass_values.append(float(design["Total Mass (kg)"]))
        cost_values.append(float(design["Total Cost (€)"]))
        r_lattice_values.append(float(design["r_lattice (mm)"]))
        t_sponge_values.append(float(design["t_sponge (mm)"]))
        support_mass_values.append(float(design["Support Mass (kg)"]))

        thickness = float(design["Thickness (mm)"])
        V_vane = float(design["V_vane (m^3)"])
        V_sponge = float(design["V_sponge (m^3)"])
        
        V_tank_mm3 = (70/1000 - 2 * thickness) ** 2 * (90/1000 - 2 * thickness)  # mm³
        V_tank = V_tank_mm3 / 10**9  # Convert to m³
        V_propellant = (V_tank - V_vane - V_sponge) * (85/90)  # m³
        V_propellant_cm3 = V_propellant * 10**6  # Convert to cm³
        eff_tank_values.append(V_propellant_cm3 * volumetric_specific_impulse_gs_per_cm3[design["Propellant"]])  # gs


    mass_norm = normalize(mass_values)
    eff_tank_norm = normalize(eff_tank_values)
    cost_norm = normalize(cost_values)
    r_lattice_norm = normalize(r_lattice_values)
    t_sponge_norm = normalize(t_sponge_values)
    support_mass_norm = normalize(support_mass_values)

    # Compute printability (equal weights of 1/3)
    printability_values = [
        (r + t + s) / 3 for r, t, s in zip(r_lattice_norm, t_sponge_norm, support_mass_norm)
    ]

    # Normalize printability
    printability_norm = normalize(printability_values)

    # Compute final trade-off score using weights
    weights = {"Mass": 4, "Effective Tank Volume": 2, "Cost": 3, "Printability": 2}
    
    trade_off_scores = [
        (4 * m + 2 * e + 3 * c + 2 * p) / sum(weights.values())
        for m, e, c, p in zip(mass_norm, eff_tank_norm, cost_norm, printability_norm)
    ]

    # Store results
    for i, design in enumerate(designs):
        design["Printability"] = printability_values[i]
        design["Trade-off Score"] = trade_off_scores[i]

    # Write updated data to a new CSV
    output_file_path = os.path.join("data", "PMD_designs_with_tradeoff.csv")
    with open(output_file_path, mode='w', newline='') as file:
        fieldnames = designs[0].keys()
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(designs)

    print(f"Trade-off analysis completed. Results saved to {output_file_path}")
    #Print top 10 designs
    designs.sort(key=lambda x: x["Trade-off Score"], reverse=True)
    print("Top 10 designs:")
    for i in range(10):
        print(f"Material: {designs[i]['Material']}, Structure Type: {designs[i]['Structure Type']}, Propellant: {designs[i]['Propellant']}, Thickness: {designs[i]['Thickness (mm)']} mm, Trade-off Score: {designs[i]['Trade-off Score']:.2f}")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    materials_file_path = os.path.join(base_dir, "data", "materials.toml")

    material_data = read_material_data(materials_file_path)

    if material_data:
        materials = material_data.get("materials", {})
        #plot_mass_vs_cost(materials)

        #plot_pressure_thickness_analysis(materials)
        #plot_stiffened_vs_unstiffened(materials)
    
    PMD_design()
    trade_off()

