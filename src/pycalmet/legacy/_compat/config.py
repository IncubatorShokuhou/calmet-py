"""Typed configuration corresponding to the legacy control-file parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


@dataclass(slots=True)
class CalwrfConfig:
    wrf_files: Sequence[str | Path]
    start: str | int
    end: str | int
    output: str | Path
    create_surface: bool = True
    compression: int = 1
    i_range: tuple[int,int] | None = None
    j_range: tuple[int,int] | None = None
    k_range: tuple[int,int] | None = None
    legacy_output: str | Path | None = None
    surface_output: str | Path | None = None
    title: str = "Create 3D.DAT file for WRF output"
    listing_output: str | Path | None = None
    legacy_file_rain_reset: bool = False


@dataclass(slots=True)
class GridConfig:
    nx: int
    ny: int
    dx_km: float
    dy_km: float
    projection: str = "LCC"
    lat0: float = 27.0
    lon0: float = 117.5
    lat1: float = 30.0
    lat2: float = 60.0
    x_origin_km: float = 0.0
    y_origin_km: float = 0.0
    false_easting_km: float = 0.0
    false_northing_km: float = 0.0
    utm_zone: int = 50
    hemisphere: str = "N"
    datum: str = "WGS-84"
    grid_type: int = 1
    ring_distances_km: Sequence[float] = ()
    ray_angles_deg: Sequence[float] = ()
    polar_processing: int = 1
    ttm_scale: float = 1.0


@dataclass(slots=True)
class TerrelSource:
    path: str | Path
    input_type: str
    input_datum: str


@dataclass(slots=True)
class TerrelConfig:
    srtm_dir: str | Path
    output: str | Path
    grid: GridConfig
    datum: str = "WGS-84"
    sampling: str = "bilinear"
    terrain_files: Sequence[str | Path] = ()
    terrain_sources: Sequence[TerrelSource] = ()
    void_fill: bool = False
    cell_radius_km: float = 0.0
    model: int = 1
    processing_method: int | None = None
    receptor_input: str | Path | None = None
    receptor_output: str | Path | None = None
    receptor_columns: int = 2
    interpolate_receptors: bool = False
    receptor_radius_km: float = 0.0
    previous_save: str | Path | None = None
    save_output: str | Path | None = None
    coastline_file: str | Path | None = None
    gshhs_file: str | Path | None = None
    coastline_source_datum: str = "WGS-84"
    coastline_wdbii_datum: str = "WGS-72"
    noise_replacement: Sequence[int] = (0, 0, 0, 0, 0)
    noise_minimum_m: Sequence[float] = (0.0, 1.0, 1.0, 1.0, 1.0)
    missing_replacement: Sequence[int] = (3, 0, 0, 0, 0)
    default_elevation_m: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0)
    plot_output: str | Path | None = None
    raw_echo_output: str | Path | None = None
    qa_hit_threshold_percent: int = 75


@dataclass(slots=True)
class CtgprocSource:
    path: str | Path
    input_type: str
    input_datum: str
    mesh_density: int = 1


@dataclass(slots=True)
class CtgprocConfig:
    landuse_files: Sequence[str | Path]
    output: str | Path
    grid: GridConfig
    category_map: dict[int, int] = field(default_factory=dict)
    input_type: str = "GLAZAS"
    mesh_density: int = 12
    input_datum: str = "ESR-S"
    output_format: str = "auto"
    coastline_file: str | Path | None = None
    gshhs_file: str | Path | None = None
    coastline_source_datum: str = "WGS-84"
    coastline_wdbii_datum: str = "WGS-72"
    coastline_swap: bool = False
    fill_empty_ocean: bool = False
    ocean_category: int = 55
    previous_landuse: str | Path | None = None
    final_output: bool = True
    landuse_sources: Sequence[CtgprocSource] = ()
    snow_processing: bool = False
    qa_cell_output: bool = False
    qa_hit_threshold_percent: int = 75
    qa_output_directory: str | Path | None = None


@dataclass(slots=True)
class MakegeoConfig:
    terrain: str | Path
    landuse: str | Path
    output: str | Path
    grid: GridConfig
    water_threshold: float = 0.5
    missing_category: int = 55
    minimum_coverage: float = 0.96
    secondary_landuse: str | Path | None = None
    use_terrain: bool = True
    input_properties: dict[int, tuple[float,float,float,float,float,float,int]] = field(default_factory=dict)
    water_categories: Sequence[int] = (51,52,53,54,55)
    output_categories: Sequence[int] = (10,20,-20,30,40,51,54,55,60,61,62,70,80,90)
    output_water_min: int = 50
    output_water_max: int = 55
    category_splits: Sequence[tuple[int,int,float]] = ()
    qa_cell: tuple[int,int] | None = None
    snow_processing: bool = False
    snow_data: str | Path | None = None
    snow_start_date: int | None = None
    snow_end_date: int | None = None
    snow_properties: dict[int, tuple[float,float,float,float,float,float,int]] = field(default_factory=dict)
    snow_age_parameters: dict[int, tuple[int,float]] = field(default_factory=dict)
    snow_roughness_method: int = 1
    snow_height_scale_m: float = 1.0
    snow_albedo_method: int = 1
    snow_depth_minimum_m: float = 0.0
    landuse_plot_output: str | Path | None = None
    terrain_plot_output: str | Path | None = None
    color_map_output: str | Path | None = None
    image_mode: int = 0
    title: str = "PyCALMET Application"


@dataclass(slots=True)
class CalmetConfig:
    meteorology: str | Path | Sequence[str | Path]
    geophysical: str | Path
    output: str | Path
    start: str
    end: str
    grid: GridConfig
    noobs: int = 2
    run_type: int = 1
    save_output: bool = True
    legacy_output_format: int = 1
    test_mode: int = 2
    regulatory_mode: int = 0
    timestep_seconds: int = 3600
    prognostic_timestep_seconds: int = 3600
    cloud_method: int = 4
    cloud_data: str | Path | None = None
    cloud_data_format: int = 2
    cloud_output: int = 0
    surface_data: str | Path | None = None
    surface_data_format: int = 2
    surface_control_stations: Sequence[tuple[str,int,float,float,float,float]] = ()
    upper_air_files: Sequence[str | Path] = ()
    upper_control_stations: Sequence[tuple[str,int,float,float,float]] = ()
    sea_files: Sequence[str | Path] = ()
    diagnostic_data: str | Path | None = None
    diagnostic_surface_temperature: int = 0
    diagnostic_lapse_rate: int = 0
    diagnostic_initial_wind: int = 0
    diagnostic_surface_wind: int = 0
    diagnostic_upper_wind: int = 0
    initial_guess_upper_station: int = -1
    initial_guess_height_range_m: Sequence[float] = (1.,1000.)
    use_initial_guess_calmet: int = 0
    initial_guess_calmet_files: Sequence[str | Path] = ()
    weight_data: str | Path | None = None
    precipitation_station_count: int = -1
    precipitation_data: str | Path | None = None
    precipitation_data_format: int = 2
    precipitation_control_stations: Sequence[tuple[str,int,float,float]] = ()
    iprog: int = 14
    irhprog: int = 1
    itprog: int = 2
    imixh: int = 1
    constb: float = 1.41
    conste: float = 0.15
    constn: float = 2400.
    constw: float = 0.16
    coriolis: float = 1.e-4
    minimum_lapse_rate: float = .001
    mixing_lapse_depth_m: float = 200.
    minimum_mixing_height_m: float = 50.
    maximum_mixing_height_m: float = 3000.
    minimum_water_mixing_height_m: float = 50.
    maximum_water_mixing_height_m: float = 3000.
    land_flux_threshold: float = 0.
    water_flux_threshold: float = .05
    water_temperature_mode: int = 0
    prognostic_ocean_category: int = 16
    overwater_flux_method: int = 10
    coastal_shelf_length_km: float = 0.
    warm_layer: int = 0
    cool_skin: int = 0
    convective_relaxation: int = 1
    convective_relaxation_time_s: float = 800.
    precipitation_interpolation: int = 2
    precipitation_radius_km: float = 100.
    precipitation_cutoff: float = .01
    turbidity_coefficient_1: float = 990.
    turbidity_coefficient_2: float = -30.
    cloudiness_coefficient_1: float = -.75
    cloudiness_coefficient_2: float = 3.4
    net_radiation_coefficient_1: float = 5.31e-13
    net_radiation_coefficient_2: float = 60.
    net_radiation_coefficient_3: float = .12
    surface_temperature_option: int = -1
    upper_temperature_option: int = -1
    lapse_height_m: float = 200.
    temperature_interpolation_power: int = 1
    temperature_radius_km: float = 500.
    temperature_station_limit: int = 5
    water_temperature_gradient_below: float = -.0098
    water_temperature_gradient_above: float = -.0045
    water_temperature_category_min: int = 55
    water_temperature_category_max: int = 55
    iavet: int = 1
    iavezi: int = 1
    mixing_wind_level: int = 1
    mnmdav: int = 1
    iwfcod: int = 1
    wind_extrapolation: int = -1
    upper_air_separation_cells: float = 4.
    extrapolate_calm: int = 0
    wind_bias: Sequence[float] = (0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
    varying_radius: bool = True
    minimum_station_radius_km: float = .1
    maximum_surface_radius_km: float = 30.
    maximum_upper_radius_km: float = 30.
    maximum_water_radius_km: float = 50.
    surface_background_radius_km: float = 1.
    upper_background_radius_km: float = 1.
    prognostic_radius_km: float = 0.
    stations_per_level: Sequence[int] = (5,5,5,5,5,5,5,5,5,5)
    wind_extrapolation_factors: Sequence[float] = (0.,0.,0.,0.,0.,0.,0.,0.,0.,0.)
    barrier_count: int = 0
    barrier_top_level: int = 0
    barriers: Sequence[tuple[float,float,float,float]] = ()
    lake_breeze_regions: Sequence[
        tuple[float,float,float,float,float,float,float,float,Sequence[int]]
    ] = ()
    ifradj: int = 1
    ikine: int = 0
    iobr: int = 0
    islope: int = 1
    critical_froude: float = 1.
    smoothing_passes: Sequence[int] = (2,4,4,4,4,4,4,4,4,4)
    divergence_limit: float = 5.e-6
    divergence_iterations: int = 50
    kinematic_alpha: float = .1
    z_faces_m: Sequence[float] = (0.,20.,40.,80.,160.,300.,600.,1000.,1500.,2200.,3000.)
    base_timezone: int = 0
    base_timezone_label: str | None = None
    terrain_radius_km: float = 15.
    half_average_angle: float = 30.
    include_w_temperature: bool = False
    compression: int = 1
    control_lines: Sequence[str] = ()
    mesopac_output: str | Path | None = None
    distance_to_coast_output: str | Path | None = None
    listing_output: str | Path | None = None
    print_fields: bool = False
    print_interval_hours: int = 1
    print_uv_levels: Sequence[int] = ()
    print_w_levels: Sequence[int] = ()
    print_temperature_levels: Sequence[int] = ()
    print_surface_fields: Sequence[int] = (0,0,0,0,0,0,0,0)
    micrometeorology_debug: bool = False
    debug_first_step: int = 1
    debug_last_step: int = 1
    wind_debug_output: int = 0
    wind_debug_levels: int = 1
    wind_debug_fields: Sequence[int] = (0,0,0,0,0,0,0,0,0)
    final_wind_debug_output: str | Path | None = None
    kinematic_wind_debug_output: str | Path | None = None
    froude_wind_debug_output: str | Path | None = None
    slope_wind_debug_output: str | Path | None = None
    wind_print_output: str | Path | None = None
