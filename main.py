import sys
import os
import toml
import math
import copy
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from ASME._Appendix13_7_a import Appendix13_7_aParams, Appendix13_7_aCalcs


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
    thickness_range = np.linspace(0.1, 5, 100) / 1000  # Thickness values in meters
    pressure_range = np.linspace(0.01, 2, 100) * 10**6  # Pressure values in Pa

    points = []
    labels = []

    for thickness in thickness_range:
        for pressure in pressure_range:
            params_inner = Appendix13_7_aParams(
                long_side_length_inside=(90 / 1000),
                short_side_length_inside=(40 / 1000),
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


def plot_pressure_thickness_analysis(materials_data):
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

    plt.suptitle("Pressure Vessel Safety Analysis for Different Materials")
    
    # Save the plot to the figures directory
    plot_path = os.path.join(figures_dir, "pressure_thickness_analysis.png")
    plt.savefig(plot_path)
    plt.show()


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    materials_file_path = os.path.join(base_dir, "data", "materials.toml")

    material_data = read_material_data(materials_file_path)

    if material_data:
        materials = material_data.get("materials", {})
        plot_pressure_thickness_analysis(materials)
