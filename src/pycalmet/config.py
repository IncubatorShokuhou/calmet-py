"""Typed configuration corresponding to the legacy control-file parameters."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Sequence

from ._path_safety import validate_calwrf_paths
from .datetime_utils import exact_integer,parse_date,parse_naive_datetime


@dataclass(slots=True)
class CalwrfConfig:
    """Configuration for converting WRF-ARW files into CALWRF meteorology.

    Attributes:
        wrf_files: WRF NetCDF files, in chronological order. Required.
        start: First naive, second-precision timestamp included in the run; accepts datetime
               or YYYYMMDDHH/ISO-like text. Required.
        end: Last naive, second-precision timestamp included in the run; accepts datetime or
             YYYYMMDDHH/ISO-like text. Required.
        output: Primary output file path. Required.
        create_surface: Create the separate legacy M2D surface file when legacy_output is
                        also requested. Defaults to ``True``.
        compression: NetCDF deflate compression level for CALWRF output (0 disables
                     compression). Defaults to ``1``.
        i_range: Inclusive one-based WRF west-east index range. Defaults to ``None``, which
                 keeps the complete west-east extent.
        j_range: Inclusive one-based WRF south-north index range. Defaults to ``None``,
                 which keeps the complete south-north extent.
        k_range: Inclusive one-based WRF vertical-level range. Defaults to ``None``, which
                 keeps every WRF vertical level.
        legacy_output: Optional traditional CALWRF 3D.DAT/M3D output generated alongside the
                       primary NetCDF file. Defaults to ``None``, which suppresses the
                       optional M3D file.
        surface_output: Optional traditional CALWRF two-dimensional surface/M2D path,
                        written when create_surface is true and legacy_output is requested.
                        Defaults to ``None``, which suppresses the M2D file even when
                        ``create_surface`` is true.
        title: Descriptive title stored in CALWRF output metadata. Defaults to ``'Create
               3D.DAT file for WRF output'``.
        listing_output: CALWRF processing log/listing destination. Defaults to ``None``,
                        which suppresses the processing listing.
    """
    wrf_files: Sequence[str | Path]
    start: datetime
    end: datetime
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

    def __post_init__(self) -> None:
        """Normalize timestamps and reject unsafe CALWRF path aliases.

        Raises:
            TypeError: If ``start`` or ``end`` is not a datetime or supported
                timestamp string.
            ValueError: If a timestamp is invalid or timezone-aware, or an
                output path aliases an input or another output.
        """
        self.start=parse_naive_datetime(self.start,'CALWRF start')
        self.end=parse_naive_datetime(self.end,'CALWRF end')
        validate_calwrf_paths(
            self.wrf_files, self.output, self.legacy_output,
            (self.surface_output
             if self.create_surface and self.legacy_output is not None else None),
            self.listing_output,
        )


@dataclass(slots=True)
class GridConfig:
    """Horizontal grid geometry and map projection shared by CALMET stages.

    Attributes:
        nx: Number of Cartesian X/east-west cells; for a TERREL polar grid, number of rings.
            Required.
        ny: Number of Cartesian Y/north-south cells; for a TERREL polar grid, number of
            rays. Required.
        dx_km: Cartesian X grid spacing in kilometres; independent of dy_km and unused for
               polar ring positions. Required.
        dy_km: Cartesian Y grid spacing in kilometres; independent of dx_km and unused for
               polar ray positions. Required.
        projection: Projection code: UTM, Transverse Mercator (TTM), Lambert conformal conic
                    (LCC), polar stereographic (PS), equatorial Mercator (EM), or Lambert
                    azimuthal equal-area (LAZA). Defaults to ``'UTM'``.
        lat0: Signed latitude of the projection origin or tangent point, in degrees (north
              positive); EM fixes it at zero. Defaults to ``27.0``.
        lon0: Signed central-meridian or tangent-point longitude, in degrees (east
              positive). Defaults to ``117.5``.
        lat1: First standard latitude for LCC, or true-scale latitude for PS. Defaults to
              ``30.0``.
        lat2: Second standard latitude for LCC; unused by PS. Defaults to ``60.0``.
        x_origin_km: Easting of the south-west corner of Cartesian cell (1,1), or the TERREL
                     polar-ring centre, in kilometres. Defaults to ``0.0``.
        y_origin_km: Northing of the south-west corner of Cartesian cell (1,1), or the
                     TERREL polar-ring centre, in kilometres. Defaults to ``0.0``.
        false_easting_km: False easting at the projection origin; used by TTM, LCC, and
                          LAZA. Defaults to ``0.0``.
        false_northing_km: False northing at the projection origin; used by TTM, LCC, and
                           LAZA. Defaults to ``0.0``.
        utm_zone: UTM zone number from 1 through 60; used only for UTM. Defaults to ``50``.
        hemisphere: UTM hemisphere, N or S; used only for UTM. Defaults to ``'N'``.
        datum: Geodetic datum or regional datum code used by the output grid. Defaults to
               ``'WGS-84'``.
        grid_type: TERREL geometry: 1 Cartesian origin at cell-(1,1) south-west corner; 2
                   Cartesian origin at its centre; 3 polar origin at ring centre. Defaults
                   to ``1``.
        ring_distances_km: TERREL polar-grid ring radii, in kilometres; supply one value per
                           ring when grid_type=3. Defaults to ``()``.
        ray_angles_deg: TERREL polar-grid ray azimuths, in degrees clockwise from north;
                        used when grid_type=3. Defaults to ``()``.
        polar_processing: TERREL-only polar method: 1 NORMAL sectors lie halfway between
                          adjacent rings/rays; 2 SCREEN sectors span the current to next
                          outer ring and halfway between rays. Defaults to ``2``.
        ttm_scale: Scale factor on the central meridian for TTM projection. Defaults to
                   ``1.0``.
    """
    nx: int
    ny: int
    dx_km: float
    dy_km: float
    projection: str = "UTM"
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
    polar_processing: int = 2
    ttm_scale: float = 1.0


@dataclass(slots=True)
class TerrelSource:
    """One terrain database supplied to TERREL.

    Attributes:
        path: Path to the source digital elevation model. Required.
        input_type: Terrain reader code, such as SRTM1, SRTM3, GTOPO30, CDED, GEN, or
                    GEOTIFF. Required.
        input_datum: Geodetic datum or regional datum code of the source coordinates.
                     Required.
    """
    path: str | Path
    input_type: str
    input_datum: str


@dataclass(slots=True)
class TerrelConfig:
    """Configuration for gridding terrain data with TERREL.

    Attributes:
        srtm_dir: Directory scanned for raw SRTM ``[NS]dd[EW]ddd.hgt`` tiles when
                  terrain_sources is empty. Required.
        output: Primary output file path. Required.
        grid: Complete grid object; an alternative to the flattened grid parameters.
              Required.
        datum: Output-grid datum/region code; individual input sources specify their own
               input datum separately. Defaults to ``'WGS-84'``.
        sampling: Terrain coordinate conversion: ``direct`` transforms every DEM sample
                  (more accurate); ``bilinear``/``legacy`` transforms sheet corners and
                  interpolates interior coordinates. Defaults to ``'bilinear'``.
        terrain_files: Selected terrain-file paths; relative SRTM tile names are resolved
                       inside srtm_dir. Defaults to ``()``.
        terrain_sources: Terrain-source records ``(path, input_type, input_datum)``;
                         input_type selects the DEM reader (for example SRTM1/3, GTOPO30,
                         CDED, GEN, or GEOTIFF). Defaults to ``()``.
        void_fill: Fill empty output terrain cells from nearby valid cells, searching no
                   farther than cell_radius_km. Defaults to ``False``.
        cell_radius_km: Maximum search radius, in kilometres, used to find valid neighbours
                        when void_fill is enabled; use at least one grid spacing. Defaults
                        to ``0.0``.
        model: TERREL output model: CALMET (1), MESOPAC (2), ISC polar (3), ISC Cartesian
               (4), NUATMOS (5), or generic (6); this also selects cell-average versus peak
               elevations. Defaults to ``1``.
        processing_method: TERREL processing method; for polar grids, 1 selects NORMAL
                           sectors and 2 SCREEN sectors, overriding grid.polar_processing.
                           Defaults to ``None``, which uses ``grid.polar_processing``.
        receptor_input: Optional discrete-receptor XY input file; supplying it enables
                        receptor processing. Defaults to ``None``, which disables
                        discrete-receptor processing.
        receptor_output: Output file containing the receptor rows with TERREL-calculated
                         terrain elevation. Defaults to ``None``, which leaves the
                         receptor-output destination unset.
        receptor_columns: Discrete-receptor input width: 2 for (X, Y), or 4 for (X, Y,
                          terrain elevation, receptor height); TERREL replaces column 3.
                          Defaults to ``4``.
        interpolate_receptors: Discrete-receptor method: interpolate elevation when true;
                               search for the peak elevation when false. Defaults to
                               ``False``.
        receptor_radius_km: Search/interpolation radius for discrete-receptor elevation, in
                            kilometres; peak-search mode requires a positive value. Defaults
                            to ``0.0``.
        previous_save: TERREL binary accumulator from an earlier pass, used to resume sample
                       sums and hit counts. Defaults to ``None``, which starts a new terrain
                       accumulation.
        save_output: Write a TERREL binary accumulator containing sample sums and hit counts
                     for a later continuation pass. Defaults to ``None``, which does not
                     write a continuation accumulator.
        coastline_file: Processed BLN coastline input; when gshhs_file is supplied, this
                        path receives the generated BLN data. Defaults to ``None``, which
                        omits processed coastline input unless another stage supplies one.
        gshhs_file: Raw GSHHS coastline database used to generate the processed BLN
                    coastline. Defaults to ``None``, which omits raw GSHHS coastline
                    processing.
        coastline_source_datum: Datum/region code assigned to coordinates read from the raw
                                GSHHS/coastline source. Defaults to ``'WGS-84'``.
        coastline_wdbii_datum: Datum/region code assigned to WDBII coastline coordinates.
                               Defaults to ``'WGS-72'``.
        noise_replacement: Per coastline class noise rule: 0 no test; 1 values below
                           noise_minimum_m become missing; 2 clamp them to the minimum; 3
                           replace them with default_elevation_m. Defaults to ``(0, 0, 0, 0,
                           0)``.
        noise_minimum_m: Minimum valid terrain elevation used by the noise tests, by
                         database class. Defaults to ``(0.0, 1.0, 1.0, 1.0, 1.0)``.
        missing_replacement: Per coastline class void rule: 0 retain; 1 replace only final
                             empty cells; 2 replace missing samples and final cells; 3 force
                             the class to default elevation. Defaults to ``(3, 0, 0, 0,
                             0)``.
        default_elevation_m: Fallback elevation used when terrain samples remain missing.
                             Defaults to ``(0.0, 0.0, 0.0, 0.0, 0.0)``.
        plot_output: Optional Surfer ASCII grid containing TERREL terrain elevations for
                     QA/plotting. Defaults to ``None``, which suppresses the Surfer terrain
                     grid.
        raw_echo_output: Optional ASCII dump of raw unfiltered terrain samples as ``(x_km,
                         y_km, elevation_m)``. Defaults to ``None``, which suppresses the
                         raw-sample dump.
        qa_hit_threshold_percent: Flag cells whose sample count is below this percentage
                                  (0–100) of the domain-average hit count. Defaults to
                                  ``75``.
    """
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
    receptor_columns: int = 4
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
    """One land-use database supplied to CTGPROC.

    Attributes:
        path: Path to the source land-use database. Required.
        input_type: Land-use reader code, such as CTG, GEN, NLCD, CORINE, GLC2K, UMDGLC, or
                    MODIS. Required.
        input_datum: Fallback datum for a source that does not carry usable
                     coordinate-reference metadata. Required.
        mesh_density: Per-side subdivision count for each source cell; each cell produces
                      ``mesh_density ** 2`` samples. Defaults to ``1``.
    """
    path: str | Path
    input_type: str
    input_datum: str
    mesh_density: int = 1


@dataclass(slots=True)
class CtgprocConfig:
    """Configuration for gridding land-use data with CTGPROC.

    Attributes:
        landuse_files: Input land-use database file paths consumed by CTGPROC. Required.
        output: Primary output file path. Required.
        grid: Complete grid object; an alternative to the flattened grid parameters.
              Required.
        category_map: Post-binning map from source to target land-use categories; hit counts
                      mapped to the same target are summed. Defaults to a new empty ``{}``
                      for each instance.
        input_type: Land-use database reader name/code, such as CTG, GEN, NLCD, CORINE,
                    GLC2K, UMDGLC, or MODIS. Defaults to ``'CTG'``.
        mesh_density: Per-side subdivision count for each input land-use cell, producing
                      mesh_density² sample points; must be at least 1. Defaults to ``1``.
        input_datum: Source datum for land-use formats without usable datum/CRS metadata;
                     self-describing GENERIC and supported GeoTIFF inputs read it from the
                     file instead. Defaults to ``'NAS-C'``.
        output_format: Output encoding: choose from the filename (auto), traditional CTGPROC
                       format (legacy), or NetCDF. Defaults to ``'auto'``.
        coastline_file: Processed BLN coastline input; when gshhs_file is supplied, this
                        path receives the generated BLN data. Defaults to ``None``, which
                        omits processed coastline input.
        gshhs_file: Raw GSHHS coastline database used to generate the processed BLN
                    coastline. Defaults to ``None``, which omits raw GSHHS coastline
                    processing.
        coastline_source_datum: Datum/region code assigned to coordinates read from the raw
                                GSHHS/coastline source. Defaults to ``'WGS-84'``.
        coastline_wdbii_datum: Datum/region code assigned to WDBII coastline coordinates.
                               Defaults to ``'WGS-72'``.
        coastline_swap: While reading samples, replace each point lying offshore of the
                        supplied coastline with ocean_category. Defaults to ``False``.
        fill_empty_ocean: On the final pass, fill zero-hit cells whose centre lies offshore
                          with ocean_category; None follows the Fortran default and enables
                          this only when coastline_file is supplied. Defaults to ``None``,
                          which enables filling exactly when ``coastline_file`` is
                          configured.
        ocean_category: Land-use category assigned to ocean cells. Defaults to ``55``.
        previous_landuse: Prior CTGPROC integer hit-count output to continue a multi-pass
                          land-use accumulation. Defaults to ``None``, which starts a new
                          land-use accumulation.
        final_output: CTGPROC pass type: true writes fractional land use for MAKEGEO; false
                      writes integer hit counts for continuation by previous_landuse.
                      Defaults to ``True``.
        landuse_sources: Land-use-source records ``(path, input_type, input_datum,
                         mesh_density)``. Defaults to ``()``.
        snow_processing: Attach SNODAS-to-land-use-grid reference records needed by MAKEGEO
                         snow processing; valid only on the final CTGPROC pass. Defaults to
                         ``False``.
        qa_cell_output: Write CTGPROC QA coordinate files for original and meshed input
                        points; output may be large. Defaults to ``False``.
        qa_hit_threshold_percent: Flag cells whose sample count is below this percentage
                                  (0–100) of the domain-average hit count. Defaults to
                                  ``75``.
        qa_output_directory: Directory receiving CTGPROC QACTG/QAGLAZ/QAMESH coordinate QA
                             files. Defaults to ``None``, which writes enabled QA files
                             beside ``output``.
    """
    landuse_files: Sequence[str | Path]
    output: str | Path
    grid: GridConfig
    category_map: dict[int, int] = field(default_factory=dict)
    input_type: str = "CTG"
    mesh_density: int = 1
    input_datum: str = "NAS-C"
    output_format: str = "auto"
    coastline_file: str | Path | None = None
    gshhs_file: str | Path | None = None
    coastline_source_datum: str = "WGS-84"
    coastline_wdbii_datum: str = "WGS-72"
    coastline_swap: bool = False
    fill_empty_ocean: bool | None = None
    ocean_category: int = 55
    previous_landuse: str | Path | None = None
    final_output: bool = True
    landuse_sources: Sequence[CtgprocSource] = ()
    snow_processing: bool = False
    qa_cell_output: bool = False
    qa_hit_threshold_percent: int = 75
    qa_output_directory: str | Path | None = None

    def __post_init__(self) -> None:
        """Resolve the automatic empty-ocean filling policy.

        A ``None`` policy follows the traditional behavior: empty offshore
        cells are filled exactly when a processed coastline is configured.
        """
        if self.fill_empty_ocean is None:
            self.fill_empty_ocean=self.coastline_file is not None


@dataclass(slots=True)
class MakegeoConfig:
    """Configuration for assembling CALMET geophysical fields with MAKEGEO.

    Attributes:
        terrain: Gridded elevation input, normally TERREL output; read only when use_terrain
                 is true. Required.
        landuse: Primary final fractional land-use data set produced by CTGPROC. Required.
        output: Primary output file path. Required.
        grid: Complete grid object; an alternative to the flattened grid parameters.
              Required.
        water_threshold: Water-category fraction required before dominant land use is
                         selected only among water categories; otherwise selection is among
                         non-water categories. Defaults to ``0.5``.
        missing_category: Category inserted when total usable land-use coverage is below
                          0.001; it must exist in input_properties. Defaults to ``55``.
        minimum_coverage: Minimum acceptable total land-use fraction: values at or above it
                          are renormalized, values below 0.001 are missing, and intermediate
                          values are errors. Defaults to ``0.96``.
        secondary_landuse: Optional secondary land-use data used only where the primary data
                           set has missing cells. Defaults to ``None``, which provides no
                           fallback for missing primary land-use cells.
        use_terrain: Read terrain elevations into GEO output when true; write zero terrain
                     when false. Defaults to ``True``.
        input_properties: Per-category ``(roughness_m, albedo, Bowen_ratio,
                          soil_heat_flux_parameter, anthropogenic_heat_W_m-2,
                          leaf_area_index, output_category)`` records; roughness uses
                          logarithmic area weighting and the other properties arithmetic
                          weighting. Defaults to a new empty ``{}`` for each instance.
        water_categories: Input land-use categories treated as water. Defaults to ``(51, 52,
                          53, 54, 55)``.
        output_categories: Ordered land-use categories retained in the geophysical output.
                           Defaults to ``(10, 20, -20, 30, 40, 51, 54, 55, 60, 61, 62, 70,
                           80, 90)``.
        output_water_min: Lowest output category regarded as water by CALMET. Defaults to
                          ``50``.
        output_water_max: Highest output category regarded as water by CALMET. Defaults to
                          ``55``.
        category_splits: MAKEGEO ``(source_category, target_category, percent)`` records
                         applied before property weighting; percentages for a source should
                         total 100. Defaults to ``()``.
        qa_cell: Optional one-based ``(i, j)`` cell for reporting intermediate terrain,
                 land-use, and surface-property calculations. Defaults to ``None``, which
                 disables per-cell intermediate-value reporting.
        snow_processing: Create one snow-adjusted GEO file per day from SNODAS depth/age
                         data over the inclusive snow_start_date..snow_end_date range.
                         Defaults to ``False``.
        snow_data: SNODAS snow-depth/age input used to build daily snow-adjusted geophysical
                   files. Defaults to ``None``, which is valid only while
                   ``snow_processing`` is disabled.
        snow_start_date: First calendar date covered by snow processing; accepts date,
                         midnight datetime, or YYYY-MM-DD/YYYYMMDD text. Defaults to
                         ``None``, which is valid only while ``snow_processing`` is
                         disabled.
        snow_end_date: Last calendar date covered by snow processing; accepts date, midnight
                       datetime, or YYYY-MM-DD/YYYYMMDD text. Defaults to ``None``, which is
                       valid only while ``snow_processing`` is disabled.
        snow_properties: Snow-covered records with the same seven fields as
                         input_properties; the mapped output category must match the normal
                         record. Defaults to a new empty ``{}`` for each instance.
        snow_age_parameters: Per snow class ``(age_time_scale_days,
                             albedo_reduction_factor)`` used by the ageing-albedo method.
                             Defaults to a new empty ``{}`` for each instance.
        snow_roughness_method: Snow roughness method: 1 reduces inferred obstacle height by
                               snow depth; 2 uses the user snow roughness wherever snow is
                               present. Defaults to ``1``.
        snow_height_scale_m: HSCL obstacle-height-to-roughness scale used by snow roughness
                             method 1. Defaults to ``10.0``.
        snow_albedo_method: Snow albedo method: 1 ages table albedo toward a reduced value;
                            2 uses user snow albedo wherever snow is present. Defaults to
                            ``1``.
        snow_depth_minimum_m: Minimum snow depth required to apply snow properties. Defaults
                              to ``0.01``.
        landuse_plot_output: Optional Surfer ASCII grid of the dominant MAKEGEO land-use
                             category. Defaults to ``None``, which suppresses this QA grid.
        terrain_plot_output: Optional Surfer ASCII grid of MAKEGEO terrain elevation.
                             Defaults to ``None``, which suppresses this QA grid.
        color_map_output: Optional Surfer colour-map companion for the land-use plot grid.
                          Defaults to ``None``, which suppresses the companion colour map.
        image_mode: QA Surfer-grid coordinate convention: cell-centre bounds (0) or
                    cell-edge bounds (1). Defaults to ``0``.
        title: GEO metadata title, limited by the traditional format to 80 characters.
               Defaults to ``'PyCALMET Application'``.
    """
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
    snow_start_date: date | None = None
    snow_end_date: date | None = None
    snow_properties: dict[int, tuple[float,float,float,float,float,float,int]] = field(default_factory=dict)
    snow_age_parameters: dict[int, tuple[int,float]] = field(default_factory=dict)
    snow_roughness_method: int = 1
    snow_height_scale_m: float = 10.0
    snow_albedo_method: int = 1
    snow_depth_minimum_m: float = 0.01
    landuse_plot_output: str | Path | None = None
    terrain_plot_output: str | Path | None = None
    color_map_output: str | Path | None = None
    image_mode: int = 0
    title: str = "PyCALMET Application"

    def __post_init__(self) -> None:
        """Normalize optional MAKEGEO snow-processing dates.

        Raises:
            TypeError: If a supplied date is not a date, datetime, or
                supported date string.
            ValueError: If a supplied datetime is timezone-aware or not at
                midnight, or if a date string is invalid.
        """
        if self.snow_start_date is not None:
            self.snow_start_date=parse_date(
                self.snow_start_date,'MAKEGEO snow_start_date')
        if self.snow_end_date is not None:
            self.snow_end_date=parse_date(
                self.snow_end_date,'MAKEGEO snow_end_date')


@dataclass(slots=True)
class CalmetConfig:
    """Configuration for the CALMET diagnostic meteorological model.

    Attributes:
        meteorology: Optional gridded prognostic meteorology input; ``iprog``
                     selects whether it supplies a Step-1 field, initial
                     guess, or pseudo-observations. ``None`` is valid only
                     for the pure-observation ``IPROG=0`` path, where
                     ``SURF.DAT`` and ``UP*.DAT`` supply the meteorology.
        geophysical: GEO geophysical grid containing terrain, land use, roughness, albedo,
                     Bowen ratio, soil/anthropogenic heat parameters, and leaf-area index.
                     Required.
        output: Primary output file path. Required.
        start: Inclusive lower boundary of the CALMET output interval ``[start, end)``;
               meteorological boundary times are selected with ``start < t <= end``, and
               each output timestamp is ``t - timestep_seconds``. Required.
        end: Exclusive upper boundary of the CALMET output interval ``[start, end)``;
             meteorological boundary times are selected with ``start < t <= end``, and each
             output timestamp is ``t - timestep_seconds``. Required.
        grid: Complete grid object; an alternative to the flattened grid parameters.
              Required.
        noobs: Observation mode: 0 uses surface/overwater/upper-air observations; 1 replaces
               upper-air observations with prognostic data; 2 uses prognostic data at all
               levels. Defaults to ``0``.
        run_type: Run type: 0 computes wind fields only; 1 also computes micrometeorology
                  and is required by CALPUFF/CALGRID. Defaults to ``1``.
        save_output: Write the gridded meteorological disk output selected by
                     legacy_output_format. Defaults to ``True``.
        legacy_output_format: Traditional disk-output type: 1 writes CALPUFF/CALGRID
                              CALMET.DAT; 2 writes MESOPUFF-II PACOUT.DAT. Defaults to
                              ``1``.
        test_mode: 1 stops after input setup checks; 2 continues into model computation.
                   Defaults to ``2``.
        regulatory_mode: Regulatory check mode: 0 disables it; 1 enforces the USEPA-required
                         mixing-height, overwater-flux, relaxation, and
                         representative-temperature choices. Defaults to ``0``.
        timestep_seconds: CALMET computational time step; it must divide evenly into one
                          hour. Defaults to ``3600``.
        prognostic_timestep_seconds: Seconds between prognostic records: a positive multiple
                                     of timestep_seconds and no more than 12 hours.
                                     Traditional PROG.DAT and native MM4.DAT inputs remain
                                     hourly by file format. Defaults to ``3600``.
        cloud_method: Cloud source: 1 surface observations (NOOBS 0/1), 2 existing
                      CLOUD.DAT, 3 prognostic 850-hPa RH, or 4 prognostic RH at all levels.
                      Defaults to ``4``.
        cloud_condensate_threshold_g_kg: Minimum layer-total non-precipitating cloud
                                         condensate (``QCLOUD + QICE``), in g kg⁻¹, used to
                                         retain prognostic cloud cover and diagnose its
                                         condensate-based ceiling. Defaults to ``0.005``.
        cloud_data: CLOUD.DAT path: input when cloud_method=2, or output destination when
                    cloud_output=1. Defaults to ``None``, which is valid unless
                    ``cloud_method=2`` or ``cloud_output=1`` requires the file.
        cloud_data_format: CLOUD.DAT format: unformatted binary (1) or formatted text (2).
                           Defaults to ``2``.
        cloud_output: Write the internally generated gridded cloud field as CLOUD.DAT (0=no,
                      1=yes); output is unavailable when cloud_method=2 reads an existing
                      file. Defaults to ``0``.
        surface_data: Hourly SURF.DAT observations for NOOBS 0/1. DIAG.DAT may replace their
                      winds; full meteorological runs still read SURF.DAT thermodynamic
                      fields. Defaults to ``None``, which omits surface observations and is
                      valid only for compatible observation modes.
        surface_data_format: SURF.DAT format: unformatted binary (1) or formatted text (2).
                             Defaults to ``2``.
        surface_control_stations: Records ``(4-char name, station ID, x_km, y_km, timezone,
                                  anemometer_height_m)`` in model coordinates; list land
                                  stations only when land/water interpolation is separate.
                                  Defaults to ``()``.
        upper_air_files: One sounding file per upper-air station for NOOBS=0. DIAG.DAT may
                         replace sounding winds; full meteorological runs still read
                         sounding temperatures. Defaults to ``()``.
        upper_control_stations: Records ``(4-char name, station ID, x_km, y_km, timezone)``,
                                one per upper-air file. Defaults to ``()``.
        sea_files: One file per overwater station; may contain water ΔT, lapse rate,
                   observed mixing height, and observed waves for negative COARE wave codes.
                   Defaults to ``()``.
        diagnostic_data: Hourly preprocessed DIAG.DAT temperatures, lapse rates, or U/V
                         winds. A file is required only when the selected observation mode
                         actually consumes an enabled diagnostic source. Defaults to
                         ``None``, which is valid while no enabled diagnostic source
                         consumes DIAG.DAT.
        diagnostic_surface_temperature: Surface-temperature source for the diagnostic wind
                                        module: compute internally (0) or read hourly
                                        preprocessed values from DIAG.DAT (1). Defaults to
                                        ``0``.
        diagnostic_lapse_rate: Domain lapse-rate source for terrain circulations: compute
                               internally from upper-air/prognostic data (0) or read hourly
                               DIAG.DAT values (1). Defaults to ``0``.
        diagnostic_initial_wind: Diagnostic-module initial-guess wind source: compute from
                                 upper/prognostic data (0) or read hourly preprocessed U/V
                                 from DIAG.DAT (1). Defaults to ``0``.
        diagnostic_surface_wind: Surface-wind source: SURF.DAT speed/direction (0) or
                                 preprocessed DIAG.DAT U/V (1). In full runs, option 1
                                 replaces only wind and retains SURF.DAT thermodynamic
                                 fields. Defaults to ``0``.
        diagnostic_upper_wind: Upper-wind source: sounding speed/direction (0) or
                               preprocessed DIAG.DAT U/V (1). In full runs, option 1
                               replaces only wind and retains sounding temperatures.
                               Defaults to ``0``.
        initial_guess_upper_station: With internally computed initial winds, 1..NUSTA
                                     selects one sounding for a uniform guess; -1 uses all
                                     soundings with 1/R² weighting for a spatially varying
                                     guess. Defaults to ``-1``.
        initial_guess_height_range_m: Bottom and top heights, in metres, used to form the
                                      diagnostic module's domain-uniform initial wind.
                                      Defaults to ``(1.0, 1000.0)``.
        use_initial_guess_calmet: Use coarse CALMET fields as the initial wind guess (0=no,
                                  1=yes); when enabled, it replaces the background wind for
                                  every IPROG mode. Defaults to ``0``.
        initial_guess_calmet_files: Coarse-grid CALMET files used as the initial wind guess
                                    when use_initial_guess_calmet=1. Defaults to ``()``.
        weight_data: WT.DAT three-dimensional observation weight Wo; the prognostic field
                     receives complementary weight 1−Wo. Defaults to ``None``, which is
                     valid unless the selected pseudo-observation mode requires WT.DAT.
        precipitation_station_count: Precipitation source: positive is the number of
                                     PRECIP.DAT stations, 0 disables precipitation input,
                                     and -1 uses prognostic precipitation. Defaults to
                                     ``0``.
        precipitation_data: PRECIP.DAT hourly station precipitation input, used when
                            precipitation_station_count is positive. Defaults to ``None``,
                            which is valid unless ``precipitation_station_count`` is
                            positive.
        precipitation_data_format: PRECIP.DAT format: unformatted binary (1) or formatted
                                   text (2). Defaults to ``2``.
        precipitation_control_stations: Records ``(4-char name, six-digit station code,
                                        x_km, y_km)``; the traditional code combines a
                                        two-digit state and four-digit station ID. Defaults
                                        to ``()``.
        iprog: Prognostic-wind use: 0 off; 1/3/13 Step-1 field (IWFCOD=0); 2/4/14 initial
               guess (IWFCOD=1); 5/15 pseudo-observations. Code families select CSUMM, MM4,
               or MM5/3D. Defaults to ``0``.
        irhprog: Relative-humidity source for NOOBS 0/1: 0 observed SURF/SEA data, 1
                 prognostic RH. NOOBS=2 always uses prognostic RH without modifying this
                 setting. Defaults to ``0``.
        itprog: Temperature source: 0 surface and upper observations (NOOBS=0); 1 surface
                observations plus prognostic aloft; 2 prognostic surface and aloft. Defaults
                to ``0``.
        imixh: Mixing-height method: 1 Maul–Carson land/water; -1 Maul–Carson land plus OCD
               water; 2 Batchvarova–Gryning land/water; -2 BG land plus OCD water. Defaults
               to ``1``.
        constb: Dimensionless B in the Venkatram neutral mechanical mixing-height relation,
                scaling friction velocity by Coriolis and free-atmosphere stability.
                Defaults to ``1.41``.
        conste: Dimensionless entrainment constant E in the modified Maul–Carson
                convective-growth equation; used when ``abs(imixh)==1``. Defaults to
                ``0.15``.
        constn: Coefficient in the stable mechanical mixing-height relation ``h = CONSTN ×
                u*^(3/2)``, evaluated with a second stability limit. Defaults to ``2400.0``.
        constw: Dimensionless overwater mechanical constant in ``h = CONSTW × u* / abs(f)``;
                an observed SEA.DAT mixing height overrides it. Defaults to ``0.16``.
        coriolis: Absolute Coriolis parameter ``abs(f)`` in s⁻¹ for mechanical mixing-height
                  equations; 999 derives it from latitude. Defaults to ``999.0``.
        minimum_lapse_rate: Floor on the stable-layer potential-temperature gradient above
                            the previous convective mixing height, in K m⁻¹. Defaults to
                            ``0.001``.
        mixing_lapse_depth_m: Depth in metres above the previous-time-step convective mixing
                              height used to calculate the stable potential-temperature
                              gradient. Defaults to ``200.0``.
        minimum_mixing_height_m: Minimum overland mixing height, in metres. Defaults to
                                 ``50.0``.
        maximum_mixing_height_m: Maximum overland mixing height, in metres. Defaults to
                                 ``3000.0``.
        minimum_water_mixing_height_m: Minimum overwater mixing height, in metres; unused
                                       where SEA.DAT supplies an observed height. Defaults
                                       to ``50.0``.
        maximum_water_mixing_height_m: Maximum overwater mixing height, in metres; unused
                                       where SEA.DAT supplies an observed height. Defaults
                                       to ``3000.0``.
        land_flux_threshold: Land heat-flux-per-boundary-layer-depth threshold, in W m⁻³;
                             convective growth requires QH greater than threshold × height.
                             Defaults to ``0.0``.
        water_flux_threshold: Overwater heat-flux-per-boundary-layer-depth threshold, in W
                              m⁻³; convective growth requires QH greater than threshold ×
                              height. Defaults to ``0.05``.
        water_temperature_mode: Overwater inputs: 0 SEA.DAT lapse rate and ΔT; 1 prognostic
                                lapse rate plus SEA.DAT ΔT (IPROG>2); 2 prognostic lapse
                                rate and ΔT (IPROG>12, 3D.DAT v2+). Defaults to ``0``.
        prognostic_ocean_category: Land-use category representing ocean in the prognostic
                                   meteorological input. Defaults to ``16``.
        overwater_flux_method: Overwater flux: 0 legacy OCD; 10 COARE Charnock/no waves; ±11
                               Oost; ±12 Taylor-Yelland. Negative codes read observed waves;
                               positive 11/12 use default waves. Defaults to ``10``.
        coastal_shelf_length_km: COARE nearshore/shallow-water roughness length scale in
                                 kilometres; zero disables this modification. Defaults to
                                 ``0.0``.
        warm_layer: COARE warm-layer correction (0=off, 1=on); keep it off for
                    infrared-radiometer SST. Defaults to ``0``.
        cool_skin: COARE cool-skin correction (0=off, 1=on); keep it off for
                   infrared-radiometer SST. Defaults to ``0``.
        convective_relaxation: Relax convective mixing height toward equilibrium in the
                               weak-positive-flux regime ``0 < QH <= threshold × height``
                               (0=no, 1=yes). Defaults to ``1``.
        convective_relaxation_time_s: E-folding relaxation time in seconds when
                                      convective_relaxation is enabled; must be at least 1
                                      s. Defaults to ``800.0``.
        precipitation_interpolation: Precipitation weighting: 1 uses 1/R, 2 uses 1/R², and 3
                                     uses a Thiessen-like exp(-R²/σ²)/R² treatment for
                                     discontinuous fields. Defaults to ``2``.
        precipitation_radius_km: For precipitation methods 1/2, station radius of influence
                                 (km); for method 3, exponential σ (km). Method 3 with zero
                                 derives half the minimum wet-to-dry-station distance.
                                 Defaults to ``100.0``.
        precipitation_cutoff: Rates below this threshold are set to zero, in mm h⁻¹.
                              Defaults to ``0.01``.
        turbidity_coefficient_1: HA1 in ``Qsw=(HA1·sinφ+HA2)(1+HB1·N^HB2)``, multiplying
                                 solar elevation; units W m⁻². Defaults to ``990.0``.
        turbidity_coefficient_2: HA2 clear-sky intercept in
                                 ``Qsw=(HA1·sinφ+HA2)(1+HB1·N^HB2)``; units W m⁻². Defaults
                                 to ``-30.0``.
        cloudiness_coefficient_1: Dimensionless HB1 cloud-attenuation amplitude in the
                                  shortwave factor ``(1+HB1·N^HB2)``. Defaults to ``-0.75``.
        cloudiness_coefficient_2: Dimensionless HB2 exponent applied to cloud fraction N in
                                  the shortwave parameterization. Defaults to ``3.4``.
        net_radiation_coefficient_1: HC1 coefficient multiplying air-temperature T⁶ in the
                                     net-radiation equation; units W m⁻² K⁻⁶. Defaults to
                                     ``5.31e-13``.
        net_radiation_coefficient_2: HC2 coefficient multiplying cloud fraction N in the
                                     net-radiation equation; units W m⁻². Defaults to
                                     ``60.0``.
        net_radiation_coefficient_3: Dimensionless HC3 denominator correction in the
                                     net-radiation equation ``(…)/(1+HC3)``. Defaults to
                                     ``0.12``.
        surface_temperature_option: When diagnostic_surface_temperature=0: positive selects
                                    a surface station, -1 a spatially varying field, and -2
                                    the domain-mean prognostic temperature. Defaults to
                                    ``-1``.
        upper_temperature_option: When diagnostic_lapse_rate=0: positive selects an
                                  upper-air station, -1 a spatially varying lapse rate, and
                                  -2 the domain-mean prognostic lapse rate. Defaults to
                                  ``-1``.
        lapse_height_m: Depth in metres used to calculate the domain-scale lapse rate for
                        diagnostic terrain-induced circulations. Defaults to ``200.0``.
        temperature_interpolation_power: Observed surface temperature/RH weighting: 1 uses
                                         1/R and 2 uses 1/R²; upper-air interpolation
                                         remains 1/R². Defaults to ``1``.
        temperature_radius_km: Radius of influence, in kilometres, for observed surface
                               temperature and relative-humidity interpolation. Defaults to
                               ``500.0``.
        temperature_station_limit: Maximum number of surface stations used in observed
                                   temperature and relative-humidity interpolation. Defaults
                                   to ``5``.
        water_temperature_gradient_below: Fallback water temperature gradient below mixing
                                          height, in K m⁻¹, when SEA.DAT does not provide
                                          one. Defaults to ``-0.0098``.
        water_temperature_gradient_above: Fallback water temperature gradient above mixing
                                          height, in K m⁻¹, when SEA.DAT does not provide
                                          one. Defaults to ``-0.0045``.
        water_temperature_category_min: First category in the inclusive range receiving
                                        separate overwater temperature/RH interpolation; set
                                        above all categories with max to disable. Defaults
                                        to ``999``.
        water_temperature_category_max: Last category in the inclusive range receiving
                                        separate overwater temperature/RH interpolation; set
                                        above all categories with min to disable. Defaults
                                        to ``999``.
        iavet: Enable spatial averaging of interpolated temperatures using the mixing-height
               averaging geometry. Defaults to ``1``.
        iavezi: Enable upwind spatial averaging of effective and convective mixing heights
                (0=no, 1=yes). Defaults to ``1``.
        mixing_wind_level: One-based wind layer (1..NZ) whose direction defines the upwind
                           averaging cone. Defaults to ``1``.
        mnmdav: Half-width, in grid cells, of the square supplementary search box used with
                the upwind cone for mixing-height and temperature averaging. Defaults to
                ``1``.
        iwfcod: Wind model: 0 objective analysis only; 1 full diagnostic module with terrain
                adjustments before objective analysis. Defaults to ``1``.
        wind_extrapolation: Vertical surface-wind extrapolation: ``abs(code)`` 1 none, 2
                            power law, 3 user factors, 4 similarity theory; negative codes
                            also ignore layer-1 upper-air data. Defaults to ``-4``.
        upper_air_separation_cells: Surface-to-upper-air proximity threshold in grid-cell
                                    spacings; a surface wind inside it is not extrapolated
                                    vertically, and -1 disables the exclusion. Defaults to
                                    ``4.0``.
        extrapolate_calm: Extrapolate a surface observation even when it reports calm wind.
                          Defaults to ``0``.
        wind_bias: Per-layer station-weight bias from -1 to 1: negative reduces upper-air
                   weight, positive reduces surface-station weight, and zero leaves 1/R²
                   weights unchanged; None derives zero for every layer. Defaults to
                   ``None``, which expands to one zero-bias value per model layer.
        varying_radius: When no valid unblocked station lies inside the relevant maximum
                        radius, use the closest valid station outside it. Defaults to
                        ``False``.
        minimum_station_radius_km: Small lower bound, in kilometres, on wind-interpolation
                                   distance; it prevents division by zero in 1/R² weighting.
                                   Defaults to ``0.1``.
        maximum_surface_radius_km: Maximum land surface-layer station radius, in kilometres;
                                   choose it to reflect terrain-limited influence. Defaults
                                   to ``30.0``.
        maximum_upper_radius_km: Maximum station radius aloft over land, in kilometres; it
                                 is generally larger than the surface radius as terrain
                                 influence decreases with height. Defaults to ``30.0``.
        maximum_water_radius_km: Maximum overwater station radius, in kilometres, used at
                                 every vertical layer over water. Defaults to ``50.0``.
        surface_background_radius_km: Distance where surface observations and the
                                      first-guess field receive equal weight. Defaults to
                                      ``1.0``.
        upper_background_radius_km: Distance where upper-air observations and the
                                    first-guess field receive equal weight. Defaults to
                                    ``1.0``.
        prognostic_radius_km: Prognostic Step-1 wind weighting distance, in kilometres; it
                              must be nonzero for IPROG 1/3/13. Defaults to ``0.0``.
        stations_per_level: Per-layer maximum number of closest stations used at a grid
                            point; None derives the Fortran default of 99 for every layer.
                            Defaults to ``None``, which expands to the traditional limit of
                            99 stations per layer.
        wind_extrapolation_factors: One multiplicative factor per layer, used only when
                                    ``abs(wind_extrapolation)==3``; None derives zero for
                                    every layer, and the required layer-1 value is unused.
                                    Defaults to ``None``, which expands to one zero factor
                                    per layer.
        barrier_count: Number of line barriers that block wind interpolation. Defaults to
                       ``0``.
        barrier_top_level: One-based highest layer (1..NZ) to which station-interpolation
                           barriers apply; None selects NZ, and it is used only when
                           barrier_count>0. Defaults to ``None``, which selects the highest
                           model layer.
        barriers: Station-wind interpolation barriers as ``(x_start, y_start, x_end,
                  y_end)`` line segments in kilometres. Defaults to ``()``.
        lake_breeze_regions: Region records containing grid-line bounds, coastline endpoints
                             in kilometres, and assigned station IDs; winds are recomputed
                             from only those stations using coast-relative 1/R² weighting.
                             Defaults to ``()``.
        ifradj: Compute Froude-number terrain-flow adjustment (0=no, 1=yes); active only
                with iwfcod=1. Defaults to ``1``.
        ikine: Compute kinematic terrain effects (0=no, 1=yes); active only with iwfcod=1.
               Defaults to ``0``.
        iobr: Apply the O'Brien procedure that forces vertical velocity to zero at the model
              top, then divergence-minimizes U/V for mass consistency (0=no, 1=yes).
              Defaults to ``0``.
        islope: Compute slope-flow parameterization (0=no, 1=yes). Defaults to ``1``.
        critical_froude: With Froude adjustment enabled, uphill flow below this critical
                         value is rotated tangent to terrain without changing speed; flow at
                         or above it passes over. Defaults to ``1.0``.
        smoothing_passes: Wind-field smoothing passes by level; None derives the Fortran
                          defaults 2 for layer 1 and 4 for every higher layer. Defaults to
                          ``None``, which selects two passes at layer 1 and four at every
                          higher layer.
        divergence_limit: Convergence target for wind-field divergence minimization.
                          Defaults to ``5e-06``.
        divergence_iterations: Maximum iterations allowed for divergence minimization.
                               Defaults to ``50``.
        kinematic_alpha: Empirical multiplier applied to the vertical gradient of
                         terrain-induced vertical velocity in the kinematic adjustment.
                         Defaults to ``0.1``.
        z_faces_m: Vertical cell-face heights in metres; supply NZ+1 values, and layer k is
                   centred at ``(face[k] + face[k+1]) / 2``. Defaults to ``(0.0, 20.0, 40.0,
                   80.0, 160.0, 300.0, 600.0, 1000.0, 1500.0, 2200.0, 3000.0)``.
        base_timezone: Base time-zone offset in CALMET's hours-west-of-UTC convention (for
                       example 5=EST; negative values are east). Defaults to ``0``.
        base_timezone_label: Modern ``UTC±HHMM`` label for the same offset represented by
                             base_timezone. Defaults to ``None``, which uses the label
                             derived from ``base_timezone`` when output metadata needs one.
        terrain_radius_km: Terrain-search radius, in kilometres, used to determine nearby
                           extrema and effective obstacle height for the Froude adjustment.
                           Defaults to ``15.0``.
        half_average_angle: Half-opening angle of the upwind spatial-averaging cone, in
                            degrees; the full opening is twice this value. Defaults to
                            ``30.0``.
        include_w_temperature: Store 3-D vertical-wind and temperature fields required by
                               CALGRID and CALPUFF's subgrid complex-terrain option.
                               Defaults to ``True``.
        compression: NetCDF deflate compression level for Python CALMET output (0 disables
                     compression). Defaults to ``1``.
        control_lines: Original CALMET control-file lines copied into legacy output
                       metadata. Defaults to ``()``.
        mesopac_output: PACOUT.DAT destination used when save_output is true and
                        legacy_output_format=2. Defaults to ``None``, which falls back to
                        ``output`` when MESOPAC output is selected.
        distance_to_coast_output: DCST.GRD destination for the distance-to-land/coast
                                  diagnostic grid. Defaults to ``None``, which suppresses
                                  the distance-to-coast grid.
        listing_output: CALMET listing/text report destination, including selected printed
                        fields and debug information. Defaults to ``None``, which uses
                        ``output`` with an ``.lst`` suffix when listing output is enabled.
        print_fields: Write selected gridded meteorological fields to listing_output every
                      print_interval_hours. Defaults to ``False``.
        print_interval_hours: Interval between printed meteorological fields, in hours.
                              Defaults to ``1``.
        print_uv_levels: Per-layer switches selecting U/V wind output when print_fields is
                         enabled. Defaults to ``()``.
        print_w_levels: NZ 0/1 switches for W at top faces 2..NZ+1 (ground-face W is zero);
                        requires print_fields and include_w_temperature. Defaults to ``()``.
        print_temperature_levels: NZ 0/1 switches for three-dimensional temperature layers;
                                  requires print_fields and include_w_temperature. Defaults
                                  to ``()``.
        print_surface_fields: Eight 0/1 print switches in this order: PGT stability,
                              friction velocity, Monin-Obukhov length, mixing height,
                              convective velocity scale, precipitation, sensible heat flux,
                              convective mixing height. Defaults to ``(0, 0, 0, 0, 0, 0, 0,
                              0)``.
        micrometeorology_debug: Print micrometeorological inputs and internal variables;
                                produces large output. Defaults to ``False``.
        debug_first_step: First time step included in micrometeorology debug output.
                          Defaults to ``1``.
        debug_last_step: Last time step included in micrometeorology debug output. Defaults
                         to ``1``.
        wind_debug_output: Master switch for writing TEST.PRT/OUT/KIN/FRD/SLP wind
                           diagnostic files (0=off, 1=on). Defaults to ``0``.
        wind_debug_levels: Number of wind-field levels written to diagnostic files, starting
                           at the surface. Defaults to ``1``.
        wind_debug_fields: Nine 0/1 switches (IPR0–IPR8): interpolated, terrain-adjusted,
                           smoothed/initial-divergence, final speed/direction, final
                           divergence, post-kinematic, post-Froude, post-slope, and
                           final-component fields. Defaults to ``(0, 0, 0, 0, 0, 0, 0, 0,
                           0)``.
        final_wind_debug_output: TEST.OUT destination for final wind diagnostic fields.
                                 Defaults to ``None``, which uses ``test.out`` beside the
                                 primary output.
        kinematic_wind_debug_output: TEST.KIN destination for wind fields after kinematic
                                     adjustment. Defaults to ``None``, which uses
                                     ``test.kin`` beside the primary output.
        froude_wind_debug_output: TEST.FRD destination for wind fields after Froude-number
                                  adjustment. Defaults to ``None``, which uses ``test.frd``
                                  beside the primary output.
        slope_wind_debug_output: TEST.SLP destination for wind fields after slope-flow
                                 adjustment. Defaults to ``None``, which uses ``test.slp``
                                 beside the primary output.
        wind_print_output: TEST.PRT destination for general wind-module diagnostic
                           variables. Defaults to ``None``, which uses ``test.prt`` beside
                           the primary output.
    """
    meteorology: str | Path | Sequence[str | Path] | None
    geophysical: str | Path
    output: str | Path
    start: datetime
    end: datetime
    grid: GridConfig
    noobs: int = 0
    run_type: int = 1
    save_output: bool = True
    legacy_output_format: int = 1
    test_mode: int = 2
    regulatory_mode: int = 0
    timestep_seconds: int = 3600
    prognostic_timestep_seconds: int = 3600
    cloud_method: int = 4
    cloud_condensate_threshold_g_kg: float = .005
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
    precipitation_station_count: int = 0
    precipitation_data: str | Path | None = None
    precipitation_data_format: int = 2
    precipitation_control_stations: Sequence[tuple[str,int,float,float]] = ()
    iprog: int = 0
    irhprog: int = 0
    itprog: int = 0
    imixh: int = 1
    constb: float = 1.41
    conste: float = 0.15
    constn: float = 2400.
    constw: float = 0.16
    coriolis: float = 999.0
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
    water_temperature_category_min: int = 999
    water_temperature_category_max: int = 999
    iavet: int = 1
    iavezi: int = 1
    mixing_wind_level: int = 1
    mnmdav: int = 1
    iwfcod: int = 1
    wind_extrapolation: int = -4
    upper_air_separation_cells: float = 4.
    extrapolate_calm: int = 0
    wind_bias: Sequence[float] | None = None
    varying_radius: bool = False
    minimum_station_radius_km: float = .1
    maximum_surface_radius_km: float = 30.
    maximum_upper_radius_km: float = 30.
    maximum_water_radius_km: float = 50.
    surface_background_radius_km: float = 1.
    upper_background_radius_km: float = 1.
    prognostic_radius_km: float = 0.
    stations_per_level: Sequence[int] | None = None
    wind_extrapolation_factors: Sequence[float] | None = None
    barrier_count: int = 0
    barrier_top_level: int | None = None
    barriers: Sequence[tuple[float,float,float,float]] = ()
    lake_breeze_regions: Sequence[
        tuple[float,float,float,float,float,float,float,float,Sequence[int]]
    ] = ()
    ifradj: int = 1
    ikine: int = 0
    iobr: int = 0
    islope: int = 1
    critical_froude: float = 1.
    smoothing_passes: Sequence[int] | None = None
    divergence_limit: float = 5.e-6
    divergence_iterations: int = 50
    kinematic_alpha: float = .1
    z_faces_m: Sequence[float] = (0.,20.,40.,80.,160.,300.,600.,1000.,1500.,2200.,3000.)
    base_timezone: int = 0
    base_timezone_label: str | None = None
    terrain_radius_km: float = 15.
    half_average_angle: float = 30.
    include_w_temperature: bool = True
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

    def __post_init__(self) -> None:
        """Normalize CALMET time fields and expand layer-dependent defaults.

        The number of model layers is ``len(z_faces_m) - 1``. ``None`` values
        for per-layer wind controls and ``barrier_top_level`` are replaced
        with their traditional CALMET defaults for that layer count.

        Raises:
            TypeError: If ``start`` or ``end`` is not a datetime or supported
                timestamp string.
            ValueError: If a timestamp is invalid or timezone-aware, or an
                integer-valued time, time-zone, or debug field is fractional
                or non-finite.
        """
        self.start=parse_naive_datetime(self.start,'CALMET start')
        self.end=parse_naive_datetime(self.end,'CALMET end')
        for name in ('timestep_seconds','prognostic_timestep_seconds',
                     'base_timezone','print_interval_hours',
                     'debug_first_step','debug_last_step'):
            setattr(self,name,exact_integer(getattr(self,name),
                                            f'CALMET {name}'))
        layers=len(self.z_faces_m)-1
        if self.wind_bias is None:
            self.wind_bias=(0.,)*layers
        if self.stations_per_level is None:
            self.stations_per_level=(99,)*layers
        if self.wind_extrapolation_factors is None:
            self.wind_extrapolation_factors=(0.,)*layers
        if self.smoothing_passes is None:
            self.smoothing_passes=(2,)+(4,)*(layers-1)
        if self.barrier_top_level is None:
            self.barrier_top_level=layers


@dataclass(slots=True)
class PipelineConfig:
    """Resolved configuration for a complete CALMET preprocessing run.

    Attributes:
        calwrf: Resolved CALWRF conversion settings. Required.
        terrel: Resolved TERREL terrain settings. Required.
        ctgproc: Resolved CTGPROC land-use settings. Required.
        makegeo: Resolved MAKEGEO geophysical settings. Required.
        calmet: Resolved CALMET meteorological-model settings. Required.
    """

    calwrf: CalwrfConfig
    terrel: TerrelConfig
    ctgproc: CtgprocConfig
    makegeo: MakegeoConfig
    calmet: CalmetConfig
