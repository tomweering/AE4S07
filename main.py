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




if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    materials_file_path = os.path.join(base_dir, "data", "materials.toml")

    material_data = read_material_data(materials_file_path)

    if material_data:
        materials = material_data.get("materials", {})
        plot_mass_vs_cost(materials)

        #plot_pressure_thickness_analysis(materials)
        #plot_stiffened_vs_unstiffened(materials)
