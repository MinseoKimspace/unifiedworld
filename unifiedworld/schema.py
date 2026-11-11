PARAM_NAMES = ("viscosity", "friction", "yield_pressure", "yield_stress", "tensile_yield_ratio")
PARAM_DIM = len(PARAM_NAMES)
MATERIAL_NAMES = ("water", "mud", "sand")
MATERIAL_APPLICABILITY = ((1,) * PARAM_DIM,) * len(MATERIAL_NAMES)
FIXED_MATERIAL_PARAMETERS = {
    "young_modulus": 1e15,
    "damping": 0.0,
    "poisson_ratio": 0.3,
    "hardening": 0.0,
    "dilatancy": 0.0,
    "hardening_rate": 1.0,
    "softening_rate": 1.0,
}
FIXED_MATERIAL_PARAMETER_UNITS = {
    "young_modulus": "Pa",
    "damping": "s",
    "poisson_ratio": "1",
    "hardening": "1",
    "dilatancy": "1",
    "hardening_rate": "1",
    "softening_rate": "1",
}
MATERIAL_FIXED_PARAMETERS = (
    {"friction": 0.0, "yield_pressure": 1e15, "yield_stress": 0.0, "tensile_yield_ratio": 1.0},
    {"friction": 0.0, "yield_pressure": 1e10, "tensile_yield_ratio": 1.0},
    {"viscosity": 0.0, "yield_pressure": 1e15, "yield_stress": 0.0, "tensile_yield_ratio": 0.0},
)


def schema_metadata():
    return {
        "id": "unifiedworld.box.linear_residual.v2",
        "parameter_names": list(PARAM_NAMES),
        "parameter_units": ["Pa*s", "1", "Pa", "Pa", "1"],
        "material_names": list(MATERIAL_NAMES),
        "fixed_material_parameters": dict(FIXED_MATERIAL_PARAMETERS),
        "fixed_material_parameter_units": dict(FIXED_MATERIAL_PARAMETER_UNITS),
        "material_fixed_parameters": {
            name: dict(values) for name, values in zip(MATERIAL_NAMES, MATERIAL_FIXED_PARAMETERS)
        },
        "units": {
            "x": "m",
            "v": "m/s",
            "mass": "kg",
            "acceleration": "m/s^2",
            "bounds": "m",
            "dt": "s",
            "spacing": "m",
            "radius": "m",
            "density": "kg/m^3",
        },
    }


def validate_schema(metadata):
    if metadata != schema_metadata():
        raise ValueError(
            "Scene/checkpoint schema, units, or fixed material parameters are incompatible"
        )


PARAM_SLICE = slice(13, 13 + PARAM_DIM)
APPLICABLE_SLICE = slice(PARAM_SLICE.stop, PARAM_SLICE.stop + PARAM_DIM)
KNOWN_SLICE = slice(APPLICABLE_SLICE.stop, APPLICABLE_SLICE.stop + PARAM_DIM)
MATERIAL_SLICE = slice(12, KNOWN_SLICE.stop)
MATERIAL_DIM = MATERIAL_SLICE.stop - MATERIAL_SLICE.start
ACCELERATION_SLICE = slice(KNOWN_SLICE.stop, KNOWN_SLICE.stop + 3)
BOUNDARY_SLICE = slice(ACCELERATION_SLICE.stop, ACCELERATION_SLICE.stop + 6)
OBSERVATION_INDEX = BOUNDARY_SLICE.stop
INPUT_DIM = OBSERVATION_INDEX + 1
