"""Generated static public wrapper for the complete CALMET pipeline.

Regenerate this file with ``python tools/generate_pipeline_api.py``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime
from pathlib import Path
from typing import Any, Final, TypeAlias

from .api import _EXPLICIT_ARGUMENTS, _capture_explicit_arguments
from .config import (
    CalmetConfig, CalwrfConfig, CtgprocConfig, CtgprocSource,
    GridConfig, MakegeoConfig, PipelineConfig, TerrelConfig,
    TerrelSource,
)

StageSpec: TypeAlias = Mapping[str, Any] | str | Path
CalwrfSpec: TypeAlias = CalwrfConfig | StageSpec
TerrelSpec: TypeAlias = TerrelConfig | StageSpec
CtgprocSpec: TypeAlias = CtgprocConfig | StageSpec
MakegeoSpec: TypeAlias = MakegeoConfig | StageSpec
CalmetSpec: TypeAlias = CalmetConfig | StageSpec

_STAGE_PARAMETERS: Final[dict[str, tuple[str, ...]]] = {
    'calwrf': ('wrf_files', 'start', 'end', 'output', 'create_surface', 'compression', 'i_range', 'j_range', 'k_range', 'legacy_output', 'surface_output', 'title', 'listing_output'),
    'terrel': ('srtm_dir', 'output', 'datum', 'sampling', 'terrain_files', 'terrain_sources', 'void_fill', 'cell_radius_km', 'model', 'processing_method', 'receptor_input', 'receptor_output', 'receptor_columns', 'interpolate_receptors', 'receptor_radius_km', 'previous_save', 'save_output', 'coastline_file', 'gshhs_file', 'coastline_source_datum', 'coastline_wdbii_datum', 'noise_replacement', 'noise_minimum_m', 'missing_replacement', 'default_elevation_m', 'plot_output', 'raw_echo_output', 'qa_hit_threshold_percent', 'grid', 'nx', 'ny', 'dx_km', 'dy_km', 'projection', 'lat0', 'lon0', 'lat1', 'lat2', 'x_origin_km', 'y_origin_km', 'false_easting_km', 'false_northing_km', 'utm_zone', 'hemisphere', 'grid_type', 'ring_distances_km', 'ray_angles_deg', 'polar_processing', 'ttm_scale'),
    'ctgproc': ('landuse_files', 'output', 'category_map', 'input_type', 'mesh_density', 'input_datum', 'output_format', 'coastline_file', 'gshhs_file', 'coastline_source_datum', 'coastline_wdbii_datum', 'coastline_swap', 'fill_empty_ocean', 'ocean_category', 'previous_landuse', 'final_output', 'landuse_sources', 'snow_processing', 'qa_cell_output', 'qa_hit_threshold_percent', 'qa_output_directory', 'grid', 'nx', 'ny', 'dx_km', 'dy_km', 'projection', 'lat0', 'lon0', 'lat1', 'lat2', 'x_origin_km', 'y_origin_km', 'false_easting_km', 'false_northing_km', 'utm_zone', 'hemisphere', 'datum', 'grid_type', 'ring_distances_km', 'ray_angles_deg', 'polar_processing', 'ttm_scale'),
    'makegeo': ('terrain', 'landuse', 'output', 'water_threshold', 'missing_category', 'minimum_coverage', 'secondary_landuse', 'use_terrain', 'input_properties', 'water_categories', 'output_categories', 'output_water_min', 'output_water_max', 'category_splits', 'qa_cell', 'snow_processing', 'snow_data', 'snow_start_date', 'snow_end_date', 'snow_properties', 'snow_age_parameters', 'snow_roughness_method', 'snow_height_scale_m', 'snow_albedo_method', 'snow_depth_minimum_m', 'landuse_plot_output', 'terrain_plot_output', 'color_map_output', 'image_mode', 'title', 'grid', 'nx', 'ny', 'dx_km', 'dy_km', 'projection', 'lat0', 'lon0', 'lat1', 'lat2', 'x_origin_km', 'y_origin_km', 'false_easting_km', 'false_northing_km', 'utm_zone', 'hemisphere', 'datum', 'grid_type', 'ring_distances_km', 'ray_angles_deg', 'polar_processing', 'ttm_scale'),
    'calmet': ('meteorology', 'geophysical', 'output', 'start', 'end', 'noobs', 'run_type', 'save_output', 'legacy_output_format', 'test_mode', 'regulatory_mode', 'timestep_seconds', 'prognostic_timestep_seconds', 'cloud_method', 'cloud_condensate_threshold_g_kg', 'cloud_data', 'cloud_data_format', 'cloud_output', 'surface_data', 'surface_data_format', 'surface_control_stations', 'upper_air_files', 'upper_control_stations', 'sea_files', 'diagnostic_data', 'diagnostic_surface_temperature', 'diagnostic_lapse_rate', 'diagnostic_initial_wind', 'diagnostic_surface_wind', 'diagnostic_upper_wind', 'initial_guess_upper_station', 'initial_guess_height_range_m', 'use_initial_guess_calmet', 'initial_guess_calmet_files', 'weight_data', 'precipitation_station_count', 'precipitation_data', 'precipitation_data_format', 'precipitation_control_stations', 'iprog', 'irhprog', 'itprog', 'imixh', 'constb', 'conste', 'constn', 'constw', 'coriolis', 'minimum_lapse_rate', 'mixing_lapse_depth_m', 'minimum_mixing_height_m', 'maximum_mixing_height_m', 'minimum_water_mixing_height_m', 'maximum_water_mixing_height_m', 'land_flux_threshold', 'water_flux_threshold', 'water_temperature_mode', 'prognostic_ocean_category', 'overwater_flux_method', 'coastal_shelf_length_km', 'warm_layer', 'cool_skin', 'convective_relaxation', 'convective_relaxation_time_s', 'precipitation_interpolation', 'precipitation_radius_km', 'precipitation_cutoff', 'turbidity_coefficient_1', 'turbidity_coefficient_2', 'cloudiness_coefficient_1', 'cloudiness_coefficient_2', 'net_radiation_coefficient_1', 'net_radiation_coefficient_2', 'net_radiation_coefficient_3', 'surface_temperature_option', 'upper_temperature_option', 'lapse_height_m', 'temperature_interpolation_power', 'temperature_radius_km', 'temperature_station_limit', 'water_temperature_gradient_below', 'water_temperature_gradient_above', 'water_temperature_category_min', 'water_temperature_category_max', 'iavet', 'iavezi', 'mixing_wind_level', 'mnmdav', 'iwfcod', 'wind_extrapolation', 'upper_air_separation_cells', 'extrapolate_calm', 'wind_bias', 'varying_radius', 'minimum_station_radius_km', 'maximum_surface_radius_km', 'maximum_upper_radius_km', 'maximum_water_radius_km', 'surface_background_radius_km', 'upper_background_radius_km', 'prognostic_radius_km', 'stations_per_level', 'wind_extrapolation_factors', 'barrier_count', 'barrier_top_level', 'barriers', 'lake_breeze_regions', 'ifradj', 'ikine', 'iobr', 'islope', 'critical_froude', 'smoothing_passes', 'divergence_limit', 'divergence_iterations', 'kinematic_alpha', 'z_faces_m', 'base_timezone', 'base_timezone_label', 'terrain_radius_km', 'half_average_angle', 'include_w_temperature', 'compression', 'control_lines', 'mesopac_output', 'distance_to_coast_output', 'listing_output', 'print_fields', 'print_interval_hours', 'print_uv_levels', 'print_w_levels', 'print_temperature_levels', 'print_surface_fields', 'micrometeorology_debug', 'debug_first_step', 'debug_last_step', 'wind_debug_output', 'wind_debug_levels', 'wind_debug_fields', 'final_wind_debug_output', 'kinematic_wind_debug_output', 'froude_wind_debug_output', 'slope_wind_debug_output', 'wind_print_output', 'grid', 'nx', 'ny', 'dx_km', 'dy_km', 'projection', 'lat0', 'lon0', 'lat1', 'lat2', 'x_origin_km', 'y_origin_km', 'false_easting_km', 'false_northing_km', 'utm_zone', 'hemisphere', 'datum', 'grid_type', 'ring_distances_km', 'ray_angles_deg', 'polar_processing', 'ttm_scale'),
}
_OMITTED_PARAMETERS: Final[dict[str, tuple[str, ...]]] = {
    'calwrf': ('end', 'output', 'start', 'wrf_files'),
    'terrel': ('dx_km', 'dy_km', 'grid', 'nx', 'ny', 'output', 'srtm_dir'),
    'ctgproc': ('category_map', 'dx_km', 'dy_km', 'grid', 'landuse_files', 'nx', 'ny', 'output'),
    'makegeo': ('dx_km', 'dy_km', 'grid', 'input_properties', 'landuse', 'nx', 'ny', 'output', 'snow_age_parameters', 'snow_properties', 'terrain'),
    'calmet': ('dx_km', 'dy_km', 'end', 'geophysical', 'grid', 'meteorology', 'nx', 'ny', 'output', 'start'),
}


@_capture_explicit_arguments
def run_pipeline(
    inp: PipelineConfig | Mapping[str, Any] | str | Path | None = None,
    *,
    calwrf: CalwrfSpec | None = None,
    terrel: TerrelSpec | None = None,
    ctgproc: CtgprocSpec | None = None,
    makegeo: MakegeoSpec | None = None,
    calmet: CalmetSpec | None = None,
    output: str | Path | None = None,
    calwrf_wrf_files: Sequence[str | Path] | None = None,
    calwrf_start: datetime | str | None = None,
    calwrf_end: datetime | str | None = None,
    calwrf_output: str | Path | None = None,
    calwrf_create_surface: bool = True,
    calwrf_compression: int = 1,
    calwrf_i_range: tuple[int, int] | None = None,
    calwrf_j_range: tuple[int, int] | None = None,
    calwrf_k_range: tuple[int, int] | None = None,
    calwrf_legacy_output: str | Path | None = None,
    calwrf_surface_output: str | Path | None = None,
    calwrf_title: str = 'Create 3D.DAT file for WRF output',
    calwrf_listing_output: str | Path | None = None,
    terrel_srtm_dir: str | Path | None = None,
    terrel_output: str | Path | None = None,
    terrel_datum: str = 'WGS-84',
    terrel_sampling: str = 'bilinear',
    terrel_terrain_files: Sequence[str | Path] = (),
    terrel_terrain_sources: Sequence[TerrelSource] = (),
    terrel_void_fill: bool = False,
    terrel_cell_radius_km: float = 0.0,
    terrel_model: int = 1,
    terrel_processing_method: int | None = None,
    terrel_receptor_input: str | Path | None = None,
    terrel_receptor_output: str | Path | None = None,
    terrel_receptor_columns: int = 4,
    terrel_interpolate_receptors: bool = False,
    terrel_receptor_radius_km: float = 0.0,
    terrel_previous_save: str | Path | None = None,
    terrel_save_output: str | Path | None = None,
    terrel_coastline_file: str | Path | None = None,
    terrel_gshhs_file: str | Path | None = None,
    terrel_coastline_source_datum: str = 'WGS-84',
    terrel_coastline_wdbii_datum: str = 'WGS-72',
    terrel_noise_replacement: Sequence[int] = (0, 0, 0, 0, 0),
    terrel_noise_minimum_m: Sequence[float] = (0.0, 1.0, 1.0, 1.0, 1.0),
    terrel_missing_replacement: Sequence[int] = (3, 0, 0, 0, 0),
    terrel_default_elevation_m: Sequence[float] = (0.0, 0.0, 0.0, 0.0, 0.0),
    terrel_plot_output: str | Path | None = None,
    terrel_raw_echo_output: str | Path | None = None,
    terrel_qa_hit_threshold_percent: int = 75,
    terrel_grid: GridConfig | Mapping[str, Any] | None = None,
    terrel_nx: int | None = None,
    terrel_ny: int | None = None,
    terrel_dx_km: float | None = None,
    terrel_dy_km: float | None = None,
    terrel_projection: str = 'UTM',
    terrel_lat0: float = 27.0,
    terrel_lon0: float = 117.5,
    terrel_lat1: float = 30.0,
    terrel_lat2: float = 60.0,
    terrel_x_origin_km: float = 0.0,
    terrel_y_origin_km: float = 0.0,
    terrel_false_easting_km: float = 0.0,
    terrel_false_northing_km: float = 0.0,
    terrel_utm_zone: int = 50,
    terrel_hemisphere: str = 'N',
    terrel_grid_type: int = 1,
    terrel_ring_distances_km: Sequence[float] = (),
    terrel_ray_angles_deg: Sequence[float] = (),
    terrel_polar_processing: int = 2,
    terrel_ttm_scale: float = 1.0,
    ctgproc_landuse_files: Sequence[str | Path] | None = None,
    ctgproc_output: str | Path | None = None,
    ctgproc_category_map: dict[int, int] | None = None,
    ctgproc_input_type: str = 'CTG',
    ctgproc_mesh_density: int = 1,
    ctgproc_input_datum: str = 'NAS-C',
    ctgproc_output_format: str = 'auto',
    ctgproc_coastline_file: str | Path | None = None,
    ctgproc_gshhs_file: str | Path | None = None,
    ctgproc_coastline_source_datum: str = 'WGS-84',
    ctgproc_coastline_wdbii_datum: str = 'WGS-72',
    ctgproc_coastline_swap: bool = False,
    ctgproc_fill_empty_ocean: bool | None = None,
    ctgproc_ocean_category: int = 55,
    ctgproc_previous_landuse: str | Path | None = None,
    ctgproc_final_output: bool = True,
    ctgproc_landuse_sources: Sequence[CtgprocSource] = (),
    ctgproc_snow_processing: bool = False,
    ctgproc_qa_cell_output: bool = False,
    ctgproc_qa_hit_threshold_percent: int = 75,
    ctgproc_qa_output_directory: str | Path | None = None,
    ctgproc_grid: GridConfig | Mapping[str, Any] | None = None,
    ctgproc_nx: int | None = None,
    ctgproc_ny: int | None = None,
    ctgproc_dx_km: float | None = None,
    ctgproc_dy_km: float | None = None,
    ctgproc_projection: str = 'UTM',
    ctgproc_lat0: float = 27.0,
    ctgproc_lon0: float = 117.5,
    ctgproc_lat1: float = 30.0,
    ctgproc_lat2: float = 60.0,
    ctgproc_x_origin_km: float = 0.0,
    ctgproc_y_origin_km: float = 0.0,
    ctgproc_false_easting_km: float = 0.0,
    ctgproc_false_northing_km: float = 0.0,
    ctgproc_utm_zone: int = 50,
    ctgproc_hemisphere: str = 'N',
    ctgproc_datum: str = 'WGS-84',
    ctgproc_grid_type: int = 1,
    ctgproc_ring_distances_km: Sequence[float] = (),
    ctgproc_ray_angles_deg: Sequence[float] = (),
    ctgproc_polar_processing: int = 2,
    ctgproc_ttm_scale: float = 1.0,
    makegeo_terrain: str | Path | None = None,
    makegeo_landuse: str | Path | None = None,
    makegeo_output: str | Path | None = None,
    makegeo_water_threshold: float = 0.5,
    makegeo_missing_category: int = 55,
    makegeo_minimum_coverage: float = 0.96,
    makegeo_secondary_landuse: str | Path | None = None,
    makegeo_use_terrain: bool = True,
    makegeo_input_properties: dict[int, tuple[float, float, float, float, float, float, int]] | None = None,
    makegeo_water_categories: Sequence[int] = (51, 52, 53, 54, 55),
    makegeo_output_categories: Sequence[int] = (10, 20, -20, 30, 40, 51, 54, 55, 60, 61, 62, 70, 80, 90),
    makegeo_output_water_min: int = 50,
    makegeo_output_water_max: int = 55,
    makegeo_category_splits: Sequence[tuple[int, int, float]] = (),
    makegeo_qa_cell: tuple[int, int] | None = None,
    makegeo_snow_processing: bool = False,
    makegeo_snow_data: str | Path | None = None,
    makegeo_snow_start_date: date | datetime | str | None = None,
    makegeo_snow_end_date: date | datetime | str | None = None,
    makegeo_snow_properties: dict[int, tuple[float, float, float, float, float, float, int]] | None = None,
    makegeo_snow_age_parameters: dict[int, tuple[int, float]] | None = None,
    makegeo_snow_roughness_method: int = 1,
    makegeo_snow_height_scale_m: float = 10.0,
    makegeo_snow_albedo_method: int = 1,
    makegeo_snow_depth_minimum_m: float = 0.01,
    makegeo_landuse_plot_output: str | Path | None = None,
    makegeo_terrain_plot_output: str | Path | None = None,
    makegeo_color_map_output: str | Path | None = None,
    makegeo_image_mode: int = 0,
    makegeo_title: str = 'PyCALMET Application',
    makegeo_grid: GridConfig | Mapping[str, Any] | None = None,
    makegeo_nx: int | None = None,
    makegeo_ny: int | None = None,
    makegeo_dx_km: float | None = None,
    makegeo_dy_km: float | None = None,
    makegeo_projection: str = 'UTM',
    makegeo_lat0: float = 27.0,
    makegeo_lon0: float = 117.5,
    makegeo_lat1: float = 30.0,
    makegeo_lat2: float = 60.0,
    makegeo_x_origin_km: float = 0.0,
    makegeo_y_origin_km: float = 0.0,
    makegeo_false_easting_km: float = 0.0,
    makegeo_false_northing_km: float = 0.0,
    makegeo_utm_zone: int = 50,
    makegeo_hemisphere: str = 'N',
    makegeo_datum: str = 'WGS-84',
    makegeo_grid_type: int = 1,
    makegeo_ring_distances_km: Sequence[float] = (),
    makegeo_ray_angles_deg: Sequence[float] = (),
    makegeo_polar_processing: int = 2,
    makegeo_ttm_scale: float = 1.0,
    calmet_meteorology: str | Path | Sequence[str | Path] | None = None,
    calmet_geophysical: str | Path | None = None,
    calmet_output: str | Path | None = None,
    calmet_start: datetime | str | None = None,
    calmet_end: datetime | str | None = None,
    calmet_noobs: int = 0,
    calmet_run_type: int = 1,
    calmet_save_output: bool = True,
    calmet_legacy_output_format: int = 1,
    calmet_test_mode: int = 2,
    calmet_regulatory_mode: int = 0,
    calmet_timestep_seconds: int = 3600,
    calmet_prognostic_timestep_seconds: int = 3600,
    calmet_cloud_method: int = 4,
    calmet_cloud_condensate_threshold_g_kg: float = 0.005,
    calmet_cloud_data: str | Path | None = None,
    calmet_cloud_data_format: int = 2,
    calmet_cloud_output: int = 0,
    calmet_surface_data: str | Path | None = None,
    calmet_surface_data_format: int = 2,
    calmet_surface_control_stations: Sequence[tuple[str, int, float, float, float, float]] = (),
    calmet_upper_air_files: Sequence[str | Path] = (),
    calmet_upper_control_stations: Sequence[tuple[str, int, float, float, float]] = (),
    calmet_sea_files: Sequence[str | Path] = (),
    calmet_diagnostic_data: str | Path | None = None,
    calmet_diagnostic_surface_temperature: int = 0,
    calmet_diagnostic_lapse_rate: int = 0,
    calmet_diagnostic_initial_wind: int = 0,
    calmet_diagnostic_surface_wind: int = 0,
    calmet_diagnostic_upper_wind: int = 0,
    calmet_initial_guess_upper_station: int = -1,
    calmet_initial_guess_height_range_m: Sequence[float] = (1.0, 1000.0),
    calmet_use_initial_guess_calmet: int = 0,
    calmet_initial_guess_calmet_files: Sequence[str | Path] = (),
    calmet_weight_data: str | Path | None = None,
    calmet_precipitation_station_count: int = 0,
    calmet_precipitation_data: str | Path | None = None,
    calmet_precipitation_data_format: int = 2,
    calmet_precipitation_control_stations: Sequence[tuple[str, int, float, float]] = (),
    calmet_iprog: int = 0,
    calmet_irhprog: int = 0,
    calmet_itprog: int = 0,
    calmet_imixh: int = 1,
    calmet_constb: float = 1.41,
    calmet_conste: float = 0.15,
    calmet_constn: float = 2400.0,
    calmet_constw: float = 0.16,
    calmet_coriolis: float = 999.0,
    calmet_minimum_lapse_rate: float = 0.001,
    calmet_mixing_lapse_depth_m: float = 200.0,
    calmet_minimum_mixing_height_m: float = 50.0,
    calmet_maximum_mixing_height_m: float = 3000.0,
    calmet_minimum_water_mixing_height_m: float = 50.0,
    calmet_maximum_water_mixing_height_m: float = 3000.0,
    calmet_land_flux_threshold: float = 0.0,
    calmet_water_flux_threshold: float = 0.05,
    calmet_water_temperature_mode: int = 0,
    calmet_prognostic_ocean_category: int = 16,
    calmet_overwater_flux_method: int = 10,
    calmet_coastal_shelf_length_km: float = 0.0,
    calmet_warm_layer: int = 0,
    calmet_cool_skin: int = 0,
    calmet_convective_relaxation: int = 1,
    calmet_convective_relaxation_time_s: float = 800.0,
    calmet_precipitation_interpolation: int = 2,
    calmet_precipitation_radius_km: float = 100.0,
    calmet_precipitation_cutoff: float = 0.01,
    calmet_turbidity_coefficient_1: float = 990.0,
    calmet_turbidity_coefficient_2: float = -30.0,
    calmet_cloudiness_coefficient_1: float = -0.75,
    calmet_cloudiness_coefficient_2: float = 3.4,
    calmet_net_radiation_coefficient_1: float = 5.31e-13,
    calmet_net_radiation_coefficient_2: float = 60.0,
    calmet_net_radiation_coefficient_3: float = 0.12,
    calmet_surface_temperature_option: int = -1,
    calmet_upper_temperature_option: int = -1,
    calmet_lapse_height_m: float = 200.0,
    calmet_temperature_interpolation_power: int = 1,
    calmet_temperature_radius_km: float = 500.0,
    calmet_temperature_station_limit: int = 5,
    calmet_water_temperature_gradient_below: float = -0.0098,
    calmet_water_temperature_gradient_above: float = -0.0045,
    calmet_water_temperature_category_min: int = 999,
    calmet_water_temperature_category_max: int = 999,
    calmet_iavet: int = 1,
    calmet_iavezi: int = 1,
    calmet_mixing_wind_level: int = 1,
    calmet_mnmdav: int = 1,
    calmet_iwfcod: int = 1,
    calmet_wind_extrapolation: int = -4,
    calmet_upper_air_separation_cells: float = 4.0,
    calmet_extrapolate_calm: int = 0,
    calmet_wind_bias: Sequence[float] | None = None,
    calmet_varying_radius: bool = False,
    calmet_minimum_station_radius_km: float = 0.1,
    calmet_maximum_surface_radius_km: float = 30.0,
    calmet_maximum_upper_radius_km: float = 30.0,
    calmet_maximum_water_radius_km: float = 50.0,
    calmet_surface_background_radius_km: float = 1.0,
    calmet_upper_background_radius_km: float = 1.0,
    calmet_prognostic_radius_km: float = 0.0,
    calmet_stations_per_level: Sequence[int] | None = None,
    calmet_wind_extrapolation_factors: Sequence[float] | None = None,
    calmet_barrier_count: int = 0,
    calmet_barrier_top_level: int | None = None,
    calmet_barriers: Sequence[tuple[float, float, float, float]] = (),
    calmet_lake_breeze_regions: Sequence[tuple[float, float, float, float, float, float, float, float, Sequence[int]]] = (),
    calmet_ifradj: int = 1,
    calmet_ikine: int = 0,
    calmet_iobr: int = 0,
    calmet_islope: int = 1,
    calmet_critical_froude: float = 1.0,
    calmet_smoothing_passes: Sequence[int] | None = None,
    calmet_divergence_limit: float = 5e-06,
    calmet_divergence_iterations: int = 50,
    calmet_kinematic_alpha: float = 0.1,
    calmet_z_faces_m: Sequence[float] = (0.0, 20.0, 40.0, 80.0, 160.0, 300.0, 600.0, 1000.0, 1500.0, 2200.0, 3000.0),
    calmet_base_timezone: int = 0,
    calmet_base_timezone_label: str | None = None,
    calmet_terrain_radius_km: float = 15.0,
    calmet_half_average_angle: float = 30.0,
    calmet_include_w_temperature: bool = True,
    calmet_compression: int = 1,
    calmet_control_lines: Sequence[str] = (),
    calmet_mesopac_output: str | Path | None = None,
    calmet_distance_to_coast_output: str | Path | None = None,
    calmet_listing_output: str | Path | None = None,
    calmet_print_fields: bool = False,
    calmet_print_interval_hours: int = 1,
    calmet_print_uv_levels: Sequence[int] = (),
    calmet_print_w_levels: Sequence[int] = (),
    calmet_print_temperature_levels: Sequence[int] = (),
    calmet_print_surface_fields: Sequence[int] = (0, 0, 0, 0, 0, 0, 0, 0),
    calmet_micrometeorology_debug: bool = False,
    calmet_debug_first_step: int = 1,
    calmet_debug_last_step: int = 1,
    calmet_wind_debug_output: int = 0,
    calmet_wind_debug_levels: int = 1,
    calmet_wind_debug_fields: Sequence[int] = (0, 0, 0, 0, 0, 0, 0, 0, 0),
    calmet_final_wind_debug_output: str | Path | None = None,
    calmet_kinematic_wind_debug_output: str | Path | None = None,
    calmet_froude_wind_debug_output: str | Path | None = None,
    calmet_slope_wind_debug_output: str | Path | None = None,
    calmet_wind_print_output: str | Path | None = None,
    calmet_grid: GridConfig | Mapping[str, Any] | None = None,
    calmet_nx: int | None = None,
    calmet_ny: int | None = None,
    calmet_dx_km: float | None = None,
    calmet_dy_km: float | None = None,
    calmet_projection: str = 'UTM',
    calmet_lat0: float = 27.0,
    calmet_lon0: float = 117.5,
    calmet_lat1: float = 30.0,
    calmet_lat2: float = 60.0,
    calmet_x_origin_km: float = 0.0,
    calmet_y_origin_km: float = 0.0,
    calmet_false_easting_km: float = 0.0,
    calmet_false_northing_km: float = 0.0,
    calmet_utm_zone: int = 50,
    calmet_hemisphere: str = 'N',
    calmet_datum: str = 'WGS-84',
    calmet_grid_type: int = 1,
    calmet_ring_distances_km: Sequence[float] = (),
    calmet_ray_angles_deg: Sequence[float] = (),
    calmet_polar_processing: int = 2,
    calmet_ttm_scale: float = 1.0,
) -> Path:
    """Run all five CALMET workflow stages without persistent intermediates.

    Args:
        inp: Pipeline configuration input. Pass a PipelineConfig or complete five-stage
             mapping by itself, a directory containing calwrf.inp, terrel.inp,
             ctgproc.inp, makegeo.inp, and calmet.inp with optional prefixed overrides,
             or None to use explicit stage specifications or prefixed typed parameters.
        calwrf: CALWRF configuration object, mapping, or control-file path. Supply this
                together with terrel, ctgproc, makegeo, and calmet when inp is None, or
                omit all five stage specifications to use prefixed parameters.
        terrel: TERREL configuration object, mapping, or control-file path.
        ctgproc: CTGPROC configuration object, mapping, or control-file path.
        makegeo: MAKEGEO configuration object, mapping, or control-file path.
        calmet: CALMET configuration object, mapping, or control-file path.
        output: Final CALMET product path; when supplied, this overrides calmet_output.
        calwrf_wrf_files: CALWRF stage. WRF NetCDF files, in chronological order.
        calwrf_start: CALWRF stage. First naive, second-precision timestamp included in
                      the run; accepts datetime or YYYYMMDDHH/ISO-like text.
        calwrf_end: CALWRF stage. Last naive, second-precision timestamp included in the
                    run; accepts datetime or YYYYMMDDHH/ISO-like text.
        calwrf_output: CALWRF stage. Primary output file path.
        calwrf_create_surface: CALWRF stage. Create the separate legacy M2D surface file
                               when legacy_output is also requested.
        calwrf_compression: CALWRF stage. NetCDF deflate compression level for CALWRF
                            output (0 disables compression).
        calwrf_i_range: CALWRF stage. Inclusive one-based WRF west-east index range.
        calwrf_j_range: CALWRF stage. Inclusive one-based WRF south-north index range.
        calwrf_k_range: CALWRF stage. Inclusive one-based WRF vertical-level range.
        calwrf_legacy_output: CALWRF stage. Optional traditional CALWRF 3D.DAT/M3D
                              output generated alongside the primary NetCDF file.
        calwrf_surface_output: CALWRF stage. Optional traditional CALWRF two-dimensional
                               surface/M2D path, written when create_surface is true and
                               legacy_output is requested.
        calwrf_title: CALWRF stage. Descriptive title stored in CALWRF output metadata.
        calwrf_listing_output: CALWRF stage. CALWRF processing log/listing destination.
        terrel_srtm_dir: TERREL stage. Directory scanned for raw SRTM
                         `[NS]dd[EW]ddd.hgt` tiles when terrain_sources is empty.
        terrel_output: TERREL stage. Primary output file path.
        terrel_datum: TERREL stage. Output-grid datum/region code; individual input
                      sources specify their own input datum separately.
        terrel_sampling: TERREL stage. Terrain coordinate conversion: `direct`
                         transforms every DEM sample (more accurate);
                         `bilinear`/`legacy` transforms sheet corners and interpolates
                         interior coordinates.
        terrel_terrain_files: TERREL stage. Selected terrain-file paths; relative SRTM
                              tile names are resolved inside srtm_dir.
        terrel_terrain_sources: TERREL stage. Terrain-source records `(path, input_type,
                                input_datum)`; input_type selects the DEM reader (for
                                example SRTM1/3, GTOPO30, CDED, GEN, or GEOTIFF).
        terrel_void_fill: TERREL stage. Fill empty output terrain cells from nearby
                          valid cells, searching no farther than cell_radius_km.
        terrel_cell_radius_km: TERREL stage. Maximum search radius, in kilometres, used
                               to find valid neighbours when void_fill is enabled; use
                               at least one grid spacing.
        terrel_model: TERREL stage. TERREL output model: CALMET (1), MESOPAC (2), ISC
                      polar (3), ISC Cartesian (4), NUATMOS (5), or generic (6); this
                      also selects cell-average versus peak elevations.
        terrel_processing_method: TERREL stage. TERREL processing method; for polar
                                  grids, 1 selects NORMAL sectors and 2 SCREEN sectors,
                                  overriding grid.polar_processing.
        terrel_receptor_input: TERREL stage. Optional discrete-receptor XY input file;
                               supplying it enables receptor processing.
        terrel_receptor_output: TERREL stage. Output file containing the receptor rows
                                with TERREL-calculated terrain elevation.
        terrel_receptor_columns: TERREL stage. Discrete-receptor input width: 2 for (X,
                                 Y), or 4 for (X, Y, terrain elevation, receptor
                                 height); TERREL replaces column 3.
        terrel_interpolate_receptors: TERREL stage. Discrete-receptor method:
                                      interpolate elevation when true; search for the
                                      peak elevation when false.
        terrel_receptor_radius_km: TERREL stage. Search/interpolation radius for
                                   discrete-receptor elevation, in kilometres;
                                   peak-search mode requires a positive value.
        terrel_previous_save: TERREL stage. TERREL binary accumulator from an earlier
                              pass, used to resume sample sums and hit counts.
        terrel_save_output: TERREL stage. Write a TERREL binary accumulator containing
                            sample sums and hit counts for a later continuation pass.
        terrel_coastline_file: TERREL stage. Processed BLN coastline input; when
                               gshhs_file is supplied, this path receives the generated
                               BLN data.
        terrel_gshhs_file: TERREL stage. Raw GSHHS coastline database used to generate
                           the processed BLN coastline.
        terrel_coastline_source_datum: TERREL stage. Datum/region code assigned to
                                       coordinates read from the raw GSHHS/coastline
                                       source.
        terrel_coastline_wdbii_datum: TERREL stage. Datum/region code assigned to WDBII
                                      coastline coordinates.
        terrel_noise_replacement: TERREL stage. Per coastline class noise rule: 0 no
                                  test; 1 values below noise_minimum_m become missing; 2
                                  clamp them to the minimum; 3 replace them with
                                  default_elevation_m.
        terrel_noise_minimum_m: TERREL stage. Minimum valid terrain elevation used by
                                the noise tests, by database class.
        terrel_missing_replacement: TERREL stage. Per coastline class void rule: 0
                                    retain; 1 replace only final empty cells; 2 replace
                                    missing samples and final cells; 3 force the class
                                    to default elevation.
        terrel_default_elevation_m: TERREL stage. Fallback elevation used when terrain
                                    samples remain missing.
        terrel_plot_output: TERREL stage. Optional Surfer ASCII grid containing TERREL
                            terrain elevations for QA/plotting.
        terrel_raw_echo_output: TERREL stage. Optional ASCII dump of raw unfiltered
                                terrain samples as `(x_km, y_km, elevation_m)`.
        terrel_qa_hit_threshold_percent: TERREL stage. Flag cells whose sample count is
                                         below this percentage (0–100) of the
                                         domain-average hit count.
        terrel_grid: TERREL stage. Complete grid object; an alternative to the flattened
                     grid parameters.
        terrel_nx: TERREL stage. Number of Cartesian X/east-west cells; for a TERREL
                   polar grid, number of rings.
        terrel_ny: TERREL stage. Number of Cartesian Y/north-south cells; for a TERREL
                   polar grid, number of rays.
        terrel_dx_km: TERREL stage. Cartesian X grid spacing in kilometres; independent
                      of dy_km and unused for polar ring positions.
        terrel_dy_km: TERREL stage. Cartesian Y grid spacing in kilometres; independent
                      of dx_km and unused for polar ray positions.
        terrel_projection: TERREL stage. Projection code: UTM, Transverse Mercator
                           (TTM), Lambert conformal conic (LCC), polar stereographic
                           (PS), equatorial Mercator (EM), or Lambert azimuthal
                           equal-area (LAZA).
        terrel_lat0: TERREL stage. Signed latitude of the projection origin or tangent
                     point, in degrees (north positive); EM fixes it at zero.
        terrel_lon0: TERREL stage. Signed central-meridian or tangent-point longitude,
                     in degrees (east positive).
        terrel_lat1: TERREL stage. First standard latitude for LCC, or true-scale
                     latitude for PS.
        terrel_lat2: TERREL stage. Second standard latitude for LCC; unused by PS.
        terrel_x_origin_km: TERREL stage. Easting of the south-west corner of Cartesian
                            cell (1,1), or the TERREL polar-ring centre, in kilometres.
        terrel_y_origin_km: TERREL stage. Northing of the south-west corner of Cartesian
                            cell (1,1), or the TERREL polar-ring centre, in kilometres.
        terrel_false_easting_km: TERREL stage. False easting at the projection origin;
                                 used by TTM, LCC, and LAZA.
        terrel_false_northing_km: TERREL stage. False northing at the projection origin;
                                  used by TTM, LCC, and LAZA.
        terrel_utm_zone: TERREL stage. UTM zone number from 1 through 60; used only for
                         UTM.
        terrel_hemisphere: TERREL stage. UTM hemisphere, N or S; used only for UTM.
        terrel_grid_type: TERREL stage. TERREL geometry: 1 Cartesian origin at
                          cell-(1,1) south-west corner; 2 Cartesian origin at its
                          centre; 3 polar origin at ring centre.
        terrel_ring_distances_km: TERREL stage. TERREL polar-grid ring radii, in
                                  kilometres; supply one value per ring when
                                  grid_type=3.
        terrel_ray_angles_deg: TERREL stage. TERREL polar-grid ray azimuths, in degrees
                               clockwise from north; used when grid_type=3.
        terrel_polar_processing: TERREL stage. TERREL-only polar method: 1 NORMAL
                                 sectors lie halfway between adjacent rings/rays; 2
                                 SCREEN sectors span the current to next outer ring and
                                 halfway between rays.
        terrel_ttm_scale: TERREL stage. Scale factor on the central meridian for TTM
                          projection.
        ctgproc_landuse_files: CTGPROC stage. Input land-use database file paths
                               consumed by CTGPROC.
        ctgproc_output: CTGPROC stage. Primary output file path.
        ctgproc_category_map: CTGPROC stage. Post-binning map from source to target
                              land-use categories; hit counts mapped to the same target
                              are summed.
        ctgproc_input_type: CTGPROC stage. Land-use database reader name/code; GLAZAS
                            selects the fixed 13000 × 12000 Asia USGS Global/GLCC-style
                            LAEA raster, while other readers include CTG, GEN, NLCD,
                            CORINE, GLC2K, UMDGLC, and MODIS.
        ctgproc_mesh_density: CTGPROC stage. Per-side subdivision count for each input
                              land-use cell, producing mesh_density² sample points; this
                              is the MESHGLAZ control for GLAZ inputs and must be at
                              least 1.
        ctgproc_input_datum: CTGPROC stage. Source datum for land-use formats without
                             usable datum/CRS metadata; headerless GLAZAS data therefore
                             requires the explicit source datum.
        ctgproc_output_format: CTGPROC stage. Output encoding: choose from the filename
                               (auto), traditional CTGPROC format (legacy), or NetCDF.
        ctgproc_coastline_file: CTGPROC stage. Processed BLN coastline input; when
                                gshhs_file is supplied, this path receives the generated
                                BLN data.
        ctgproc_gshhs_file: CTGPROC stage. Raw GSHHS coastline database used to generate
                            the processed BLN coastline.
        ctgproc_coastline_source_datum: CTGPROC stage. Datum/region code assigned to
                                        coordinates read from the raw GSHHS/coastline
                                        source.
        ctgproc_coastline_wdbii_datum: CTGPROC stage. Datum/region code assigned to
                                       WDBII coastline coordinates.
        ctgproc_coastline_swap: CTGPROC stage. While reading samples, replace each point
                                lying offshore of the supplied coastline with
                                ocean_category.
        ctgproc_fill_empty_ocean: CTGPROC stage. On the final pass, fill zero-hit cells
                                  whose centre lies offshore with ocean_category; None
                                  follows the Fortran default and enables this only when
                                  coastline_file is supplied.
        ctgproc_ocean_category: CTGPROC stage. Land-use category assigned to ocean
                                cells.
        ctgproc_previous_landuse: CTGPROC stage. Prior CTGPROC integer hit-count output
                                  to continue a multi-pass land-use accumulation.
        ctgproc_final_output: CTGPROC stage. CTGPROC pass type: true writes fractional
                              land use for MAKEGEO; false writes integer hit counts for
                              continuation by previous_landuse.
        ctgproc_landuse_sources: CTGPROC stage. Land-use-source records `(path,
                                 input_type, input_datum, mesh_density)`.
        ctgproc_snow_processing: CTGPROC stage. Attach SNODAS-to-land-use-grid reference
                                 records needed by MAKEGEO snow processing; valid only
                                 on the final CTGPROC pass.
        ctgproc_qa_cell_output: CTGPROC stage. Write CTGPROC QA coordinate files for
                                original and meshed input points; output may be large.
        ctgproc_qa_hit_threshold_percent: CTGPROC stage. Flag cells whose sample count
                                          is below this percentage (0–100) of the
                                          domain-average hit count.
        ctgproc_qa_output_directory: CTGPROC stage. Directory receiving CTGPROC
                                     QACTG/QAGLAZ/QAMESH coordinate QA files.
        ctgproc_grid: CTGPROC stage. Complete grid object; an alternative to the
                      flattened grid parameters.
        ctgproc_nx: CTGPROC stage. Number of Cartesian X/east-west cells; for a TERREL
                    polar grid, number of rings.
        ctgproc_ny: CTGPROC stage. Number of Cartesian Y/north-south cells; for a TERREL
                    polar grid, number of rays.
        ctgproc_dx_km: CTGPROC stage. Cartesian X grid spacing in kilometres;
                       independent of dy_km and unused for polar ring positions.
        ctgproc_dy_km: CTGPROC stage. Cartesian Y grid spacing in kilometres;
                       independent of dx_km and unused for polar ray positions.
        ctgproc_projection: CTGPROC stage. Projection code: UTM, Transverse Mercator
                            (TTM), Lambert conformal conic (LCC), polar stereographic
                            (PS), equatorial Mercator (EM), or Lambert azimuthal
                            equal-area (LAZA).
        ctgproc_lat0: CTGPROC stage. Signed latitude of the projection origin or tangent
                      point, in degrees (north positive); EM fixes it at zero.
        ctgproc_lon0: CTGPROC stage. Signed central-meridian or tangent-point longitude,
                      in degrees (east positive).
        ctgproc_lat1: CTGPROC stage. First standard latitude for LCC, or true-scale
                      latitude for PS.
        ctgproc_lat2: CTGPROC stage. Second standard latitude for LCC; unused by PS.
        ctgproc_x_origin_km: CTGPROC stage. Easting of the south-west corner of
                             Cartesian cell (1,1), or the TERREL polar-ring centre, in
                             kilometres.
        ctgproc_y_origin_km: CTGPROC stage. Northing of the south-west corner of
                             Cartesian cell (1,1), or the TERREL polar-ring centre, in
                             kilometres.
        ctgproc_false_easting_km: CTGPROC stage. False easting at the projection origin;
                                  used by TTM, LCC, and LAZA.
        ctgproc_false_northing_km: CTGPROC stage. False northing at the projection
                                   origin; used by TTM, LCC, and LAZA.
        ctgproc_utm_zone: CTGPROC stage. UTM zone number from 1 through 60; used only
                          for UTM.
        ctgproc_hemisphere: CTGPROC stage. UTM hemisphere, N or S; used only for UTM.
        ctgproc_datum: CTGPROC stage. Output-grid datum/region code; individual input
                       sources specify their own input datum separately.
        ctgproc_grid_type: CTGPROC stage. TERREL geometry: 1 Cartesian origin at
                           cell-(1,1) south-west corner; 2 Cartesian origin at its
                           centre; 3 polar origin at ring centre.
        ctgproc_ring_distances_km: CTGPROC stage. TERREL polar-grid ring radii, in
                                   kilometres; supply one value per ring when
                                   grid_type=3.
        ctgproc_ray_angles_deg: CTGPROC stage. TERREL polar-grid ray azimuths, in
                                degrees clockwise from north; used when grid_type=3.
        ctgproc_polar_processing: CTGPROC stage. TERREL-only polar method: 1 NORMAL
                                  sectors lie halfway between adjacent rings/rays; 2
                                  SCREEN sectors span the current to next outer ring and
                                  halfway between rays.
        ctgproc_ttm_scale: CTGPROC stage. Scale factor on the central meridian for TTM
                           projection.
        makegeo_terrain: MAKEGEO stage. Gridded elevation input, normally TERREL output;
                         read only when use_terrain is true.
        makegeo_landuse: MAKEGEO stage. Primary final fractional land-use data set
                         produced by CTGPROC.
        makegeo_output: MAKEGEO stage. Primary output file path.
        makegeo_water_threshold: MAKEGEO stage. Water-category fraction required before
                                 dominant land use is selected only among water
                                 categories; otherwise selection is among non-water
                                 categories.
        makegeo_missing_category: MAKEGEO stage. Category inserted when total usable
                                  land-use coverage is below 0.001; it must exist in
                                  input_properties.
        makegeo_minimum_coverage: MAKEGEO stage. Minimum acceptable total land-use
                                  fraction: values at or above it are renormalized,
                                  values below 0.001 are missing, and intermediate
                                  values are errors.
        makegeo_secondary_landuse: MAKEGEO stage. Optional secondary land-use data used
                                   only where the primary data set has missing cells.
        makegeo_use_terrain: MAKEGEO stage. Read terrain elevations into GEO output when
                             true; write zero terrain when false.
        makegeo_input_properties: MAKEGEO stage. Per-category `(roughness_m, albedo,
                                  Bowen_ratio, soil_heat_flux_parameter,
                                  anthropogenic_heat_W_m-2, leaf_area_index,
                                  output_category)` records; roughness uses logarithmic
                                  area weighting and the other properties arithmetic
                                  weighting.
        makegeo_water_categories: MAKEGEO stage. Input land-use categories treated as
                                  water.
        makegeo_output_categories: MAKEGEO stage. Ordered land-use categories retained
                                   in the geophysical output.
        makegeo_output_water_min: MAKEGEO stage. Lowest output category regarded as
                                  water by CALMET.
        makegeo_output_water_max: MAKEGEO stage. Highest output category regarded as
                                  water by CALMET.
        makegeo_category_splits: MAKEGEO stage. MAKEGEO `(source_category,
                                 target_category, percent)` records applied before
                                 property weighting; percentages for a source should
                                 total 100.
        makegeo_qa_cell: MAKEGEO stage. Optional one-based `(i, j)` cell for reporting
                         intermediate terrain, land-use, and surface-property
                         calculations.
        makegeo_snow_processing: MAKEGEO stage. Create one snow-adjusted GEO file per
                                 day from SNODAS depth/age data over the inclusive
                                 snow_start_date..snow_end_date range.
        makegeo_snow_data: MAKEGEO stage. SNODAS snow-depth/age input used to build
                           daily snow-adjusted geophysical files.
        makegeo_snow_start_date: MAKEGEO stage. First calendar date covered by snow
                                 processing; accepts date, midnight datetime, or
                                 YYYY-MM-DD/YYYYMMDD text.
        makegeo_snow_end_date: MAKEGEO stage. Last calendar date covered by snow
                               processing; accepts date, midnight datetime, or
                               YYYY-MM-DD/YYYYMMDD text.
        makegeo_snow_properties: MAKEGEO stage. Snow-covered records with the same seven
                                 fields as input_properties; the mapped output category
                                 must match the normal record.
        makegeo_snow_age_parameters: MAKEGEO stage. Per snow class
                                     `(age_time_scale_days, albedo_reduction_factor)`
                                     used by the ageing-albedo method.
        makegeo_snow_roughness_method: MAKEGEO stage. Snow roughness method: 1 reduces
                                       inferred obstacle height by snow depth; 2 uses
                                       the user snow roughness wherever snow is present.
        makegeo_snow_height_scale_m: MAKEGEO stage. HSCL obstacle-height-to-roughness
                                     scale used by snow roughness method 1.
        makegeo_snow_albedo_method: MAKEGEO stage. Snow albedo method: 1 ages table
                                    albedo toward a reduced value; 2 uses user snow
                                    albedo wherever snow is present.
        makegeo_snow_depth_minimum_m: MAKEGEO stage. Minimum snow depth required to
                                      apply snow properties.
        makegeo_landuse_plot_output: MAKEGEO stage. Optional Surfer ASCII grid of the
                                     dominant MAKEGEO land-use category.
        makegeo_terrain_plot_output: MAKEGEO stage. Optional Surfer ASCII grid of
                                     MAKEGEO terrain elevation.
        makegeo_color_map_output: MAKEGEO stage. Optional Surfer colour-map companion
                                  for the land-use plot grid.
        makegeo_image_mode: MAKEGEO stage. QA Surfer-grid coordinate convention:
                            cell-centre bounds (0) or cell-edge bounds (1).
        makegeo_title: MAKEGEO stage. GEO metadata title, limited by the traditional
                       format to 80 characters.
        makegeo_grid: MAKEGEO stage. Complete grid object; an alternative to the
                      flattened grid parameters.
        makegeo_nx: MAKEGEO stage. Number of Cartesian X/east-west cells; for a TERREL
                    polar grid, number of rings.
        makegeo_ny: MAKEGEO stage. Number of Cartesian Y/north-south cells; for a TERREL
                    polar grid, number of rays.
        makegeo_dx_km: MAKEGEO stage. Cartesian X grid spacing in kilometres;
                       independent of dy_km and unused for polar ring positions.
        makegeo_dy_km: MAKEGEO stage. Cartesian Y grid spacing in kilometres;
                       independent of dx_km and unused for polar ray positions.
        makegeo_projection: MAKEGEO stage. Projection code: UTM, Transverse Mercator
                            (TTM), Lambert conformal conic (LCC), polar stereographic
                            (PS), equatorial Mercator (EM), or Lambert azimuthal
                            equal-area (LAZA).
        makegeo_lat0: MAKEGEO stage. Signed latitude of the projection origin or tangent
                      point, in degrees (north positive); EM fixes it at zero.
        makegeo_lon0: MAKEGEO stage. Signed central-meridian or tangent-point longitude,
                      in degrees (east positive).
        makegeo_lat1: MAKEGEO stage. First standard latitude for LCC, or true-scale
                      latitude for PS.
        makegeo_lat2: MAKEGEO stage. Second standard latitude for LCC; unused by PS.
        makegeo_x_origin_km: MAKEGEO stage. Easting of the south-west corner of
                             Cartesian cell (1,1), or the TERREL polar-ring centre, in
                             kilometres.
        makegeo_y_origin_km: MAKEGEO stage. Northing of the south-west corner of
                             Cartesian cell (1,1), or the TERREL polar-ring centre, in
                             kilometres.
        makegeo_false_easting_km: MAKEGEO stage. False easting at the projection origin;
                                  used by TTM, LCC, and LAZA.
        makegeo_false_northing_km: MAKEGEO stage. False northing at the projection
                                   origin; used by TTM, LCC, and LAZA.
        makegeo_utm_zone: MAKEGEO stage. UTM zone number from 1 through 60; used only
                          for UTM.
        makegeo_hemisphere: MAKEGEO stage. UTM hemisphere, N or S; used only for UTM.
        makegeo_datum: MAKEGEO stage. Output-grid datum/region code; individual input
                       sources specify their own input datum separately.
        makegeo_grid_type: MAKEGEO stage. TERREL geometry: 1 Cartesian origin at
                           cell-(1,1) south-west corner; 2 Cartesian origin at its
                           centre; 3 polar origin at ring centre.
        makegeo_ring_distances_km: MAKEGEO stage. TERREL polar-grid ring radii, in
                                   kilometres; supply one value per ring when
                                   grid_type=3.
        makegeo_ray_angles_deg: MAKEGEO stage. TERREL polar-grid ray azimuths, in
                                degrees clockwise from north; used when grid_type=3.
        makegeo_polar_processing: MAKEGEO stage. TERREL-only polar method: 1 NORMAL
                                  sectors lie halfway between adjacent rings/rays; 2
                                  SCREEN sectors span the current to next outer ring and
                                  halfway between rays.
        makegeo_ttm_scale: MAKEGEO stage. Scale factor on the central meridian for TTM
                           projection.
        calmet_meteorology: CALMET stage. Gridded prognostic meteorology input; iprog
                            selects whether it supplies a Step-1 field, initial guess,
                            or pseudo-observations.
        calmet_geophysical: CALMET stage. GEO geophysical grid containing terrain, land
                            use, roughness, albedo, Bowen ratio, soil/anthropogenic heat
                            parameters, and leaf-area index.
        calmet_output: CALMET stage. Primary output file path.
        calmet_start: CALMET stage. Inclusive lower boundary of the CALMET output
                      interval `[start, end)`; meteorological boundary times are
                      selected with `start < t <= end`, and each output timestamp is `t
                      - timestep_seconds`.
        calmet_end: CALMET stage. Exclusive upper boundary of the CALMET output interval
                    `[start, end)`; meteorological boundary times are selected with
                    `start < t <= end`, and each output timestamp is `t -
                    timestep_seconds`.
        calmet_noobs: CALMET stage. Observation mode: 0 uses surface/overwater/upper-air
                      observations; 1 replaces upper-air observations with prognostic
                      data; 2 uses prognostic data at all levels.
        calmet_run_type: CALMET stage. Run type: 0 computes wind fields only; 1 also
                         computes micrometeorology and is required by CALPUFF/CALGRID.
        calmet_save_output: CALMET stage. Write the gridded meteorological disk output
                            selected by legacy_output_format.
        calmet_legacy_output_format: CALMET stage. Traditional disk-output type: 1
                                     writes CALPUFF/CALGRID CALMET.DAT; 2 writes
                                     MESOPUFF-II PACOUT.DAT.
        calmet_test_mode: CALMET stage. 1 stops after input setup checks; 2 continues
                          into model computation.
        calmet_regulatory_mode: CALMET stage. Regulatory check mode: 0 disables it; 1
                                enforces the USEPA-required mixing-height,
                                overwater-flux, relaxation, and
                                representative-temperature choices.
        calmet_timestep_seconds: CALMET stage. CALMET computational time step; it must
                                 divide evenly into one hour.
        calmet_prognostic_timestep_seconds: CALMET stage. Seconds between prognostic
                                            records: a positive multiple of
                                            timestep_seconds and no more than 12 hours.
                                            Traditional PROG.DAT and native MM4.DAT
                                            inputs remain hourly by file format.
        calmet_cloud_method: CALMET stage. Cloud source: 1 surface observations (NOOBS
                             0/1), 2 existing CLOUD.DAT, 3 prognostic 850-hPa RH, or 4
                             prognostic RH at all levels.
        calmet_cloud_condensate_threshold_g_kg: CALMET stage. Minimum layer-total
                                                non-precipitating cloud condensate
                                                (`QCLOUD + QICE`), in g kg⁻¹, used to
                                                retain prognostic cloud cover and
                                                diagnose its condensate-based ceiling.
        calmet_cloud_data: CALMET stage. CLOUD.DAT path: input when cloud_method=2, or
                           output destination when cloud_output=1.
        calmet_cloud_data_format: CALMET stage. CLOUD.DAT format: unformatted binary (1)
                                  or formatted text (2).
        calmet_cloud_output: CALMET stage. Write the internally generated gridded cloud
                             field as CLOUD.DAT (0=no, 1=yes); output is unavailable
                             when cloud_method=2 reads an existing file.
        calmet_surface_data: CALMET stage. Hourly SURF.DAT observations for NOOBS 0/1.
                             DIAG.DAT may replace their winds; full meteorological runs
                             still read SURF.DAT thermodynamic fields.
        calmet_surface_data_format: CALMET stage. SURF.DAT format: unformatted binary
                                    (1) or formatted text (2).
        calmet_surface_control_stations: CALMET stage. Records `(4-char name, station
                                         ID, x_km, y_km, timezone, anemometer_height_m)`
                                         in model coordinates; list land stations only
                                         when land/water interpolation is separate.
        calmet_upper_air_files: CALMET stage. One sounding file per upper-air station
                                for NOOBS=0. DIAG.DAT may replace sounding winds; full
                                meteorological runs still read sounding temperatures.
        calmet_upper_control_stations: CALMET stage. Records `(4-char name, station ID,
                                       x_km, y_km, timezone)`, one per upper-air file.
        calmet_sea_files: CALMET stage. One file per overwater station; may contain
                          water ΔT, lapse rate, observed mixing height, and observed
                          waves for negative COARE wave codes.
        calmet_diagnostic_data: CALMET stage. Hourly preprocessed DIAG.DAT temperatures,
                                lapse rates, or U/V winds. A file is required only when
                                the selected observation mode actually consumes an
                                enabled diagnostic source.
        calmet_diagnostic_surface_temperature: CALMET stage. Surface-temperature source
                                               for the diagnostic wind module: compute
                                               internally (0) or read hourly
                                               preprocessed values from DIAG.DAT (1).
        calmet_diagnostic_lapse_rate: CALMET stage. Domain lapse-rate source for terrain
                                      circulations: compute internally from
                                      upper-air/prognostic data (0) or read hourly
                                      DIAG.DAT values (1).
        calmet_diagnostic_initial_wind: CALMET stage. Diagnostic-module initial-guess
                                        wind source: compute from upper/prognostic data
                                        (0) or read hourly preprocessed U/V from
                                        DIAG.DAT (1).
        calmet_diagnostic_surface_wind: CALMET stage. Surface-wind source: SURF.DAT
                                        speed/direction (0) or preprocessed DIAG.DAT U/V
                                        (1). In full runs, option 1 replaces only wind
                                        and retains SURF.DAT thermodynamic fields.
        calmet_diagnostic_upper_wind: CALMET stage. Upper-wind source: sounding
                                      speed/direction (0) or preprocessed DIAG.DAT U/V
                                      (1). In full runs, option 1 replaces only wind and
                                      retains sounding temperatures.
        calmet_initial_guess_upper_station: CALMET stage. With internally computed
                                            initial winds, 1..NUSTA selects one sounding
                                            for a uniform guess; -1 uses all soundings
                                            with 1/R² weighting for a spatially varying
                                            guess.
        calmet_initial_guess_height_range_m: CALMET stage. Bottom and top heights, in
                                             metres, used to form the diagnostic
                                             module's domain-uniform initial wind.
        calmet_use_initial_guess_calmet: CALMET stage. Use coarse CALMET fields as the
                                         initial wind guess (0=no, 1=yes); when enabled,
                                         it replaces the background wind for every IPROG
                                         mode.
        calmet_initial_guess_calmet_files: CALMET stage. Coarse-grid CALMET files used
                                           as the initial wind guess when
                                           use_initial_guess_calmet=1.
        calmet_weight_data: CALMET stage. WT.DAT three-dimensional observation weight
                            Wo; the prognostic field receives complementary weight 1−Wo.
        calmet_precipitation_station_count: CALMET stage. Precipitation source: positive
                                            is the number of PRECIP.DAT stations, 0
                                            disables precipitation input, and -1 uses
                                            prognostic precipitation.
        calmet_precipitation_data: CALMET stage. PRECIP.DAT hourly station precipitation
                                   input, used when precipitation_station_count is
                                   positive.
        calmet_precipitation_data_format: CALMET stage. PRECIP.DAT format: unformatted
                                          binary (1) or formatted text (2).
        calmet_precipitation_control_stations: CALMET stage. Records `(4-char name,
                                               six-digit station code, x_km, y_km)`; the
                                               traditional code combines a two-digit
                                               state and four-digit station ID.
        calmet_iprog: CALMET stage. Prognostic-wind use: 0 off; 1/3/13 Step-1 field
                      (IWFCOD=0); 2/4/14 initial guess (IWFCOD=1); 5/15
                      pseudo-observations. Code families select CSUMM, MM4, or MM5/3D.
        calmet_irhprog: CALMET stage. Relative-humidity source for NOOBS 0/1: 0 observed
                        SURF/SEA data, 1 prognostic RH. NOOBS=2 always uses prognostic
                        RH without modifying this setting.
        calmet_itprog: CALMET stage. Temperature source: 0 surface and upper
                       observations (NOOBS=0); 1 surface observations plus prognostic
                       aloft; 2 prognostic surface and aloft.
        calmet_imixh: CALMET stage. Mixing-height method: 1 Maul–Carson land/water; -1
                      Maul–Carson land plus OCD water; 2 Batchvarova–Gryning land/water;
                      -2 BG land plus OCD water.
        calmet_constb: CALMET stage. Dimensionless B in the Venkatram neutral mechanical
                       mixing-height relation, scaling friction velocity by Coriolis and
                       free-atmosphere stability.
        calmet_conste: CALMET stage. Dimensionless entrainment constant E in the
                       modified Maul–Carson convective-growth equation; used when
                       `abs(imixh)==1`.
        calmet_constn: CALMET stage. Coefficient in the stable mechanical mixing-height
                       relation `h = CONSTN × u*^(3/2)`, evaluated with a second
                       stability limit.
        calmet_constw: CALMET stage. Dimensionless overwater mechanical constant in `h =
                       CONSTW × u* / abs(f)`; an observed SEA.DAT mixing height
                       overrides it.
        calmet_coriolis: CALMET stage. Absolute Coriolis parameter `abs(f)` in s⁻¹ for
                         mechanical mixing-height equations; 999 derives it from
                         latitude.
        calmet_minimum_lapse_rate: CALMET stage. Floor on the stable-layer
                                   potential-temperature gradient above the previous
                                   convective mixing height, in K m⁻¹.
        calmet_mixing_lapse_depth_m: CALMET stage. Depth in metres above the
                                     previous-time-step convective mixing height used to
                                     calculate the stable potential-temperature
                                     gradient.
        calmet_minimum_mixing_height_m: CALMET stage. Minimum overland mixing height, in
                                        metres.
        calmet_maximum_mixing_height_m: CALMET stage. Maximum overland mixing height, in
                                        metres.
        calmet_minimum_water_mixing_height_m: CALMET stage. Minimum overwater mixing
                                              height, in metres; unused where SEA.DAT
                                              supplies an observed height.
        calmet_maximum_water_mixing_height_m: CALMET stage. Maximum overwater mixing
                                              height, in metres; unused where SEA.DAT
                                              supplies an observed height.
        calmet_land_flux_threshold: CALMET stage. Land
                                    heat-flux-per-boundary-layer-depth threshold, in W
                                    m⁻³; convective growth requires QH greater than
                                    threshold × height.
        calmet_water_flux_threshold: CALMET stage. Overwater
                                     heat-flux-per-boundary-layer-depth threshold, in W
                                     m⁻³; convective growth requires QH greater than
                                     threshold × height.
        calmet_water_temperature_mode: CALMET stage. Overwater inputs: 0 SEA.DAT lapse
                                       rate and ΔT; 1 prognostic lapse rate plus SEA.DAT
                                       ΔT (IPROG>2); 2 prognostic lapse rate and ΔT
                                       (IPROG>12, 3D.DAT v2+).
        calmet_prognostic_ocean_category: CALMET stage. Land-use category representing
                                          ocean in the prognostic meteorological input.
        calmet_overwater_flux_method: CALMET stage. Overwater flux: 0 legacy OCD; 10
                                      COARE Charnock/no waves; ±11 Oost; ±12
                                      Taylor-Yelland. Negative codes read observed
                                      waves; positive 11/12 use default waves.
        calmet_coastal_shelf_length_km: CALMET stage. COARE nearshore/shallow-water
                                        roughness length scale in kilometres; zero
                                        disables this modification.
        calmet_warm_layer: CALMET stage. COARE warm-layer correction (0=off, 1=on); keep
                           it off for infrared-radiometer SST.
        calmet_cool_skin: CALMET stage. COARE cool-skin correction (0=off, 1=on); keep
                          it off for infrared-radiometer SST.
        calmet_convective_relaxation: CALMET stage. Relax convective mixing height
                                      toward equilibrium in the weak-positive-flux
                                      regime `0 < QH <= threshold × height` (0=no,
                                      1=yes).
        calmet_convective_relaxation_time_s: CALMET stage. E-folding relaxation time in
                                             seconds when convective_relaxation is
                                             enabled; must be at least 1 s.
        calmet_precipitation_interpolation: CALMET stage. Precipitation weighting: 1
                                            uses 1/R, 2 uses 1/R², and 3 uses a
                                            Thiessen-like exp(-R²/σ²)/R² treatment for
                                            discontinuous fields.
        calmet_precipitation_radius_km: CALMET stage. For precipitation methods 1/2,
                                        station radius of influence (km); for method 3,
                                        exponential σ (km). Method 3 with zero derives
                                        half the minimum wet-to-dry-station distance.
        calmet_precipitation_cutoff: CALMET stage. Rates below this threshold are set to
                                     zero, in mm h⁻¹.
        calmet_turbidity_coefficient_1: CALMET stage. HA1 in
                                        `Qsw=(HA1·sinφ+HA2)(1+HB1·N^HB2)`, multiplying
                                        solar elevation; units W m⁻².
        calmet_turbidity_coefficient_2: CALMET stage. HA2 clear-sky intercept in
                                        `Qsw=(HA1·sinφ+HA2)(1+HB1·N^HB2)`; units W m⁻².
        calmet_cloudiness_coefficient_1: CALMET stage. Dimensionless HB1
                                         cloud-attenuation amplitude in the shortwave
                                         factor `(1+HB1·N^HB2)`.
        calmet_cloudiness_coefficient_2: CALMET stage. Dimensionless HB2 exponent
                                         applied to cloud fraction N in the shortwave
                                         parameterization.
        calmet_net_radiation_coefficient_1: CALMET stage. HC1 coefficient multiplying
                                            air-temperature T⁶ in the net-radiation
                                            equation; units W m⁻² K⁻⁶.
        calmet_net_radiation_coefficient_2: CALMET stage. HC2 coefficient multiplying
                                            cloud fraction N in the net-radiation
                                            equation; units W m⁻².
        calmet_net_radiation_coefficient_3: CALMET stage. Dimensionless HC3 denominator
                                            correction in the net-radiation equation
                                            `(…)/(1+HC3)`.
        calmet_surface_temperature_option: CALMET stage. When
                                           diagnostic_surface_temperature=0: positive
                                           selects a surface station, -1 a spatially
                                           varying field, and -2 the domain-mean
                                           prognostic temperature.
        calmet_upper_temperature_option: CALMET stage. When diagnostic_lapse_rate=0:
                                         positive selects an upper-air station, -1 a
                                         spatially varying lapse rate, and -2 the
                                         domain-mean prognostic lapse rate.
        calmet_lapse_height_m: CALMET stage. Depth in metres used to calculate the
                               domain-scale lapse rate for diagnostic terrain-induced
                               circulations.
        calmet_temperature_interpolation_power: CALMET stage. Observed surface
                                                temperature/RH weighting: 1 uses 1/R and
                                                2 uses 1/R²; upper-air interpolation
                                                remains 1/R².
        calmet_temperature_radius_km: CALMET stage. Radius of influence, in kilometres,
                                      for observed surface temperature and
                                      relative-humidity interpolation.
        calmet_temperature_station_limit: CALMET stage. Maximum number of surface
                                          stations used in observed temperature and
                                          relative-humidity interpolation.
        calmet_water_temperature_gradient_below: CALMET stage. Fallback water
                                                 temperature gradient below mixing
                                                 height, in K m⁻¹, when SEA.DAT does not
                                                 provide one.
        calmet_water_temperature_gradient_above: CALMET stage. Fallback water
                                                 temperature gradient above mixing
                                                 height, in K m⁻¹, when SEA.DAT does not
                                                 provide one.
        calmet_water_temperature_category_min: CALMET stage. First category in the
                                               inclusive range receiving separate
                                               overwater temperature/RH interpolation;
                                               set above all categories with max to
                                               disable.
        calmet_water_temperature_category_max: CALMET stage. Last category in the
                                               inclusive range receiving separate
                                               overwater temperature/RH interpolation;
                                               set above all categories with min to
                                               disable.
        calmet_iavet: CALMET stage. Enable spatial averaging of interpolated
                      temperatures using the mixing-height averaging geometry.
        calmet_iavezi: CALMET stage. Enable upwind spatial averaging of effective and
                       convective mixing heights (0=no, 1=yes).
        calmet_mixing_wind_level: CALMET stage. One-based wind layer (1..NZ) whose
                                  direction defines the upwind averaging cone.
        calmet_mnmdav: CALMET stage. Half-width, in grid cells, of the square
                       supplementary search box used with the upwind cone for
                       mixing-height and temperature averaging.
        calmet_iwfcod: CALMET stage. Wind model: 0 objective analysis only; 1 full
                       diagnostic module with terrain adjustments before objective
                       analysis.
        calmet_wind_extrapolation: CALMET stage. Vertical surface-wind extrapolation:
                                   `abs(code)` 1 none, 2 power law, 3 user factors, 4
                                   similarity theory; negative codes also ignore layer-1
                                   upper-air data.
        calmet_upper_air_separation_cells: CALMET stage. Surface-to-upper-air proximity
                                           threshold in grid-cell spacings; a surface
                                           wind inside it is not extrapolated
                                           vertically, and -1 disables the exclusion.
        calmet_extrapolate_calm: CALMET stage. Extrapolate a surface observation even
                                 when it reports calm wind.
        calmet_wind_bias: CALMET stage. Per-layer station-weight bias from -1 to 1:
                          negative reduces upper-air weight, positive reduces
                          surface-station weight, and zero leaves 1/R² weights
                          unchanged; None derives zero for every layer.
        calmet_varying_radius: CALMET stage. When no valid unblocked station lies inside
                               the relevant maximum radius, use the closest valid
                               station outside it.
        calmet_minimum_station_radius_km: CALMET stage. Small lower bound, in
                                          kilometres, on wind-interpolation distance; it
                                          prevents division by zero in 1/R² weighting.
        calmet_maximum_surface_radius_km: CALMET stage. Maximum land surface-layer
                                          station radius, in kilometres; choose it to
                                          reflect terrain-limited influence.
        calmet_maximum_upper_radius_km: CALMET stage. Maximum station radius aloft over
                                        land, in kilometres; it is generally larger than
                                        the surface radius as terrain influence
                                        decreases with height.
        calmet_maximum_water_radius_km: CALMET stage. Maximum overwater station radius,
                                        in kilometres, used at every vertical layer over
                                        water.
        calmet_surface_background_radius_km: CALMET stage. Distance where surface
                                             observations and the first-guess field
                                             receive equal weight.
        calmet_upper_background_radius_km: CALMET stage. Distance where upper-air
                                           observations and the first-guess field
                                           receive equal weight.
        calmet_prognostic_radius_km: CALMET stage. Prognostic Step-1 wind weighting
                                     distance, in kilometres; it must be nonzero for
                                     IPROG 1/3/13.
        calmet_stations_per_level: CALMET stage. Per-layer maximum number of closest
                                   stations used at a grid point; None derives the
                                   Fortran default of 99 for every layer.
        calmet_wind_extrapolation_factors: CALMET stage. One multiplicative factor per
                                           layer, used only when
                                           `abs(wind_extrapolation)==3`; None derives
                                           zero for every layer, and the required
                                           layer-1 value is unused.
        calmet_barrier_count: CALMET stage. Number of line barriers that block wind
                              interpolation.
        calmet_barrier_top_level: CALMET stage. One-based highest layer (1..NZ) to which
                                  station-interpolation barriers apply; None selects NZ,
                                  and it is used only when barrier_count>0.
        calmet_barriers: CALMET stage. Station-wind interpolation barriers as `(x_start,
                         y_start, x_end, y_end)` line segments in kilometres.
        calmet_lake_breeze_regions: CALMET stage. Region records containing grid-line
                                    bounds, coastline endpoints in kilometres, and
                                    assigned station IDs; winds are recomputed from only
                                    those stations using coast-relative 1/R² weighting.
        calmet_ifradj: CALMET stage. Compute Froude-number terrain-flow adjustment
                       (0=no, 1=yes); active only with iwfcod=1.
        calmet_ikine: CALMET stage. Compute kinematic terrain effects (0=no, 1=yes);
                      active only with iwfcod=1.
        calmet_iobr: CALMET stage. Apply the O'Brien procedure that forces vertical
                     velocity to zero at the model top, then divergence-minimizes U/V
                     for mass consistency (0=no, 1=yes).
        calmet_islope: CALMET stage. Compute slope-flow parameterization (0=no, 1=yes).
        calmet_critical_froude: CALMET stage. With Froude adjustment enabled, uphill
                                flow below this critical value is rotated tangent to
                                terrain without changing speed; flow at or above it
                                passes over.
        calmet_smoothing_passes: CALMET stage. Wind-field smoothing passes by level;
                                 None derives the Fortran defaults 2 for layer 1 and 4
                                 for every higher layer.
        calmet_divergence_limit: CALMET stage. Convergence target for wind-field
                                 divergence minimization.
        calmet_divergence_iterations: CALMET stage. Maximum iterations allowed for
                                      divergence minimization.
        calmet_kinematic_alpha: CALMET stage. Empirical multiplier applied to the
                                vertical gradient of terrain-induced vertical velocity
                                in the kinematic adjustment.
        calmet_z_faces_m: CALMET stage. Vertical cell-face heights in metres; supply
                          NZ+1 values, and layer k is centred at `(face[k] + face[k+1])
                          / 2`.
        calmet_base_timezone: CALMET stage. Base time-zone offset in CALMET's
                              hours-west-of-UTC convention (for example 5=EST; negative
                              values are east).
        calmet_base_timezone_label: CALMET stage. Modern `UTC±HHMM` label for the same
                                    offset represented by base_timezone.
        calmet_terrain_radius_km: CALMET stage. Terrain-search radius, in kilometres,
                                  used to determine nearby extrema and effective
                                  obstacle height for the Froude adjustment.
        calmet_half_average_angle: CALMET stage. Half-opening angle of the upwind
                                   spatial-averaging cone, in degrees; the full opening
                                   is twice this value.
        calmet_include_w_temperature: CALMET stage. Store 3-D vertical-wind and
                                      temperature fields required by CALGRID and
                                      CALPUFF's subgrid complex-terrain option.
        calmet_compression: CALMET stage. NetCDF deflate compression level for Python
                            CALMET output (0 disables compression).
        calmet_control_lines: CALMET stage. Original CALMET control-file lines copied
                              into legacy output metadata.
        calmet_mesopac_output: CALMET stage. PACOUT.DAT destination used when
                               save_output is true and legacy_output_format=2.
        calmet_distance_to_coast_output: CALMET stage. DCST.GRD destination for the
                                         distance-to-land/coast diagnostic grid.
        calmet_listing_output: CALMET stage. CALMET listing/text report destination,
                               including selected printed fields and debug information.
        calmet_print_fields: CALMET stage. Write selected gridded meteorological fields
                             to listing_output every print_interval_hours.
        calmet_print_interval_hours: CALMET stage. Interval between printed
                                     meteorological fields, in hours.
        calmet_print_uv_levels: CALMET stage. Per-layer switches selecting U/V wind
                                output when print_fields is enabled.
        calmet_print_w_levels: CALMET stage. NZ 0/1 switches for W at top faces 2..NZ+1
                               (ground-face W is zero); requires print_fields and
                               include_w_temperature.
        calmet_print_temperature_levels: CALMET stage. NZ 0/1 switches for
                                         three-dimensional temperature layers; requires
                                         print_fields and include_w_temperature.
        calmet_print_surface_fields: CALMET stage. Eight 0/1 print switches in this
                                     order: PGT stability, friction velocity,
                                     Monin-Obukhov length, mixing height, convective
                                     velocity scale, precipitation, sensible heat flux,
                                     convective mixing height.
        calmet_micrometeorology_debug: CALMET stage. Print micrometeorological inputs
                                       and internal variables; produces large output.
        calmet_debug_first_step: CALMET stage. First time step included in
                                 micrometeorology debug output.
        calmet_debug_last_step: CALMET stage. Last time step included in
                                micrometeorology debug output.
        calmet_wind_debug_output: CALMET stage. Master switch for writing
                                  TEST.PRT/OUT/KIN/FRD/SLP wind diagnostic files (0=off,
                                  1=on).
        calmet_wind_debug_levels: CALMET stage. Number of wind-field levels written to
                                  diagnostic files, starting at the surface.
        calmet_wind_debug_fields: CALMET stage. Nine 0/1 switches (IPR0–IPR8):
                                  interpolated, terrain-adjusted,
                                  smoothed/initial-divergence, final speed/direction,
                                  final divergence, post-kinematic, post-Froude,
                                  post-slope, and final-component fields.
        calmet_final_wind_debug_output: CALMET stage. TEST.OUT destination for final
                                        wind diagnostic fields.
        calmet_kinematic_wind_debug_output: CALMET stage. TEST.KIN destination for wind
                                            fields after kinematic adjustment.
        calmet_froude_wind_debug_output: CALMET stage. TEST.FRD destination for wind
                                         fields after Froude-number adjustment.
        calmet_slope_wind_debug_output: CALMET stage. TEST.SLP destination for wind
                                        fields after slope-flow adjustment.
        calmet_wind_print_output: CALMET stage. TEST.PRT destination for general
                                  wind-module diagnostic variables.
        calmet_grid: CALMET stage. Complete grid object; an alternative to the flattened
                     grid parameters.
        calmet_nx: CALMET stage. Number of Cartesian X/east-west cells; for a TERREL
                   polar grid, number of rings.
        calmet_ny: CALMET stage. Number of Cartesian Y/north-south cells; for a TERREL
                   polar grid, number of rays.
        calmet_dx_km: CALMET stage. Cartesian X grid spacing in kilometres; independent
                      of dy_km and unused for polar ring positions.
        calmet_dy_km: CALMET stage. Cartesian Y grid spacing in kilometres; independent
                      of dx_km and unused for polar ray positions.
        calmet_projection: CALMET stage. Projection code: UTM, Transverse Mercator
                           (TTM), Lambert conformal conic (LCC), polar stereographic
                           (PS), equatorial Mercator (EM), or Lambert azimuthal
                           equal-area (LAZA).
        calmet_lat0: CALMET stage. Signed latitude of the projection origin or tangent
                     point, in degrees (north positive); EM fixes it at zero.
        calmet_lon0: CALMET stage. Signed central-meridian or tangent-point longitude,
                     in degrees (east positive).
        calmet_lat1: CALMET stage. First standard latitude for LCC, or true-scale
                     latitude for PS.
        calmet_lat2: CALMET stage. Second standard latitude for LCC; unused by PS.
        calmet_x_origin_km: CALMET stage. Easting of the south-west corner of Cartesian
                            cell (1,1), or the TERREL polar-ring centre, in kilometres.
        calmet_y_origin_km: CALMET stage. Northing of the south-west corner of Cartesian
                            cell (1,1), or the TERREL polar-ring centre, in kilometres.
        calmet_false_easting_km: CALMET stage. False easting at the projection origin;
                                 used by TTM, LCC, and LAZA.
        calmet_false_northing_km: CALMET stage. False northing at the projection origin;
                                  used by TTM, LCC, and LAZA.
        calmet_utm_zone: CALMET stage. UTM zone number from 1 through 60; used only for
                         UTM.
        calmet_hemisphere: CALMET stage. UTM hemisphere, N or S; used only for UTM.
        calmet_datum: CALMET stage. Output-grid datum/region code; individual input
                      sources specify their own input datum separately.
        calmet_grid_type: CALMET stage. TERREL geometry: 1 Cartesian origin at
                          cell-(1,1) south-west corner; 2 Cartesian origin at its
                          centre; 3 polar origin at ring centre.
        calmet_ring_distances_km: CALMET stage. TERREL polar-grid ring radii, in
                                  kilometres; supply one value per ring when
                                  grid_type=3.
        calmet_ray_angles_deg: CALMET stage. TERREL polar-grid ray azimuths, in degrees
                               clockwise from north; used when grid_type=3.
        calmet_polar_processing: CALMET stage. TERREL-only polar method: 1 NORMAL
                                 sectors lie halfway between adjacent rings/rays; 2
                                 SCREEN sectors span the current to next outer ring and
                                 halfway between rays.
        calmet_ttm_scale: CALMET stage. Scale factor on the central meridian for TTM
                          projection.

    Returns:
        Path to the final CALMET meteorological output.

    Raises:
        TypeError: If pipeline input modes are combined, a stage specification
            is missing, or a parameter is unknown, duplicated, or invalid.
        ValueError: If a stage configuration fails validation.
        OSError: If a required input cannot be read or the final output cannot
            be written.
    """
    values: dict[str, Any] = locals()
    supplied: frozenset[str] = _EXPLICIT_ARGUMENTS.get() or frozenset()
    parameter_only: bool = (
        inp is None
        and calwrf is None
        and terrel is None
        and ctgproc is None
        and makegeo is None
        and calmet is None
    )
    from .pipeline import _run_pipeline, _select_stage_parameters

    stage_parameters: dict[str, dict[str, Any]] = {
        stage: _select_stage_parameters(
            stage, values, supplied, _STAGE_PARAMETERS[stage],
            _OMITTED_PARAMETERS[stage], parameter_only=parameter_only,
        )
        for stage in _STAGE_PARAMETERS
    }
    return _run_pipeline(
        inp,
        calwrf=calwrf,
        terrel=terrel,
        ctgproc=ctgproc,
        makegeo=makegeo,
        calmet=calmet,
        output=output,
        stage_parameters=stage_parameters,
    )


__all__: Final[list[str]] = ["run_pipeline"]
