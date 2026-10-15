from io import StringIO
from pathlib import Path

import pytest

import numpy as np
import tifffile
from netCDF4 import Dataset

from pycalmet._terrel import (_3cd_origin,_accumulate_3cd,_accumulate_arm3,
                              _accumulate_dmdf,_accumulate_generic,_accumulate_geotiff,
                              _accumulate_gtopo30,_accumulate_nzgen,
                              _accumulate_points,
                              _accumulate_usgs30,_accumulate_usgs_degree_dem,
                              _accumulate_usgsla,
                              _cell_average,_cell_averages,_cell_search_count,_cell_type_coordinates,_dem_records,_geotiff_geometry,_gtopo_spec,
                              _fill_polar_angular_gaps,
                              _finalize_terrain,
                              _fixed_peak_radius,_grid_domain_metres,
                              _fill_void_cells,_fortran_records,_interp2d,_load_save,
                              _parse_dem_header,_save_header,
                              _polar_cell_indices,
                              _read_receptors,
                              _terrain_filter,
                              _hit_statistics,
                              _polar_boundaries,
                              _parse_dem_profiles,_read_arm3,
                              _read_dmdf,_read_generic,_tile_origin,
                              _usgsla_spec,_write_legacy,_write_legacy_header,_write_netcdf,
                              _write_plot_output,_write_save,run,
                              _write_surfer_grid,
                              UsgslaSpec)
from pycalmet.config import GridConfig,TerrelConfig,TerrelSource


def test_terrel_netcdf_carries_makegeo_grid_contract(tmp_path):
    grid=GridConfig(2,1,1.5,1.5,x_origin_km=4.,y_origin_km=5.,
                    projection='TTM',ttm_scale=.9996)
    path=tmp_path/'terrain.nc'
    _write_netcdf(path,np.ones((1,2)),np.ones(2,dtype=np.int64),grid,grid.datum)
    with Dataset(path) as dataset:
        assert (dataset.grid_nx,dataset.grid_ny)==(2,1)
        assert (dataset.grid_dx_km,dataset.grid_dy_km)==(1.5,1.5)
        assert (dataset.x_origin_km,dataset.y_origin_km)==(4.,5.)
        assert dataset.ttm_scale==pytest.approx(.9996)


@pytest.mark.parametrize('projection',('UTM','LCC','TTM','PS','EM','LAZA'))
def test_terrel_legacy_header_formats_hemispheres_for_all_projections(projection):
    grid=GridConfig(1,1,1.,1.,projection=projection,lat0=-27.,lon0=-117.5,
                    lat1=-30.,lat2=-60.,utm_zone=12,hemisphere='S')
    stream=StringIO(); _write_legacy_header(stream,grid,grid.datum)
    lines=stream.getvalue().splitlines(); projection_index=4
    assert len(lines[0])==96
    assert lines[projection_index].strip()==projection
    parameters=lines[projection_index+1]
    if projection=='UTM':
        assert parameters=='  12S   '
    else:
        assert parameters[:16].strip()=='27S'
        assert parameters[16:32].strip()=='117.5W'
    datum_index=projection_index+2+(projection in {'LCC','LAZA','TTM'})
    assert lines[datum_index]=='WGS-84  02-21-2003  '


@pytest.mark.parametrize(
    ("name", "origin"),
    [("N26E117.HGT", (26, 117)), ("S03W042.hgt", (-3, -42))],
)
def test_srtm_name_origin(name, origin):
    assert _tile_origin(Path(name)) == origin


def test_hit_statistics_match_terrel_nonempty_average_and_nint_thresholds():
    # Average excludes zero-hit cells: (2 + 5 + 8) / 3 = 5.
    assert _hit_statistics([0,2,5,8],75)==(5,4,6,1,1)


def test_hit_statistics_reject_invalid_threshold():
    with pytest.raises(ValueError,match='ITHRES'):
        _hit_statistics([1],101)


def test_non_srtm_name_is_rejected():
    with pytest.raises(ValueError, match="not an SRTM"):
        _tile_origin(Path("terrain.bin"))


def test_control_tile_list_filters_directory_case_insensitively(tmp_path,monkeypatch):
    (tmp_path/'n26e117.hgt').write_bytes(b'placeholder')
    (tmp_path/'N26E118.HGT').write_bytes(b'placeholder')
    seen=[]
    def fake_accumulate(path,grid,sums,counts,**kwargs):
        seen.append(path.name); sums[0]+=10.; counts[0]+=1; return 1
    monkeypatch.setattr('pycalmet._terrel._accumulate_tile',fake_accumulate)
    config=TerrelConfig(tmp_path,tmp_path/'terrain.nc',GridConfig(1,1,1.,1.),
                        terrain_files=['N26E117.HGT'])
    result=run(config)
    assert seen==['n26e117.hgt'] and result.tiles==1 and result.populated_cells==1


def test_gtopo30_filename_selects_standard_and_antarctic_layouts():
    assert _gtopo_spec(Path('W180N90.DEM'))== (90,-180,6000,4800,120)
    assert _gtopo_spec(Path('E000S60.DEM'))== (-60,0,3600,7200,120)


def test_gtopo30_accumulates_big_endian_samples_with_legacy_interpolation(tmp_path,monkeypatch):
    path=tmp_path/'E000N01.DEM'
    np.array([[10,20],[30,-32768]],dtype='>i2').tofile(path)
    monkeypatch.setattr('pycalmet._terrel.project_geographic',
                        lambda lon,lat,grid,datum:(lon*1000.,lat*1000.))
    grid=GridConfig(2,2,.5,.5,y_origin_km=.5)
    sums=np.zeros(4); counts=np.zeros(4,dtype=np.int64)
    inside=_accumulate_gtopo30(path,grid,sums,counts,spec=(1,0,2,2,2))
    assert inside==3
    np.testing.assert_array_equal(counts,[1,0,1,1])
    np.testing.assert_allclose(sums,[30,0,10,20])


def test_gtopo30_msheet1_projects_every_sample_coordinate(tmp_path, monkeypatch):
    path = tmp_path / "E000N01.DEM"
    np.array([[10, 20], [30, 40]], dtype=">i2").tofile(path)
    calls = []

    def project(longitude, latitude, grid, datum):
        calls.append((np.asarray(longitude).copy(), np.asarray(latitude).copy()))
        return np.asarray(longitude) * 1000.0, np.asarray(latitude) * 1000.0

    monkeypatch.setattr("pycalmet._terrel.project_geographic", project)
    grid = GridConfig(4, 4, 0.5, 0.5)
    sums = np.zeros(16)
    counts = np.zeros(16, dtype=np.int64)
    _accumulate_gtopo30(
        path, grid, sums, counts, spec=(1, 0, 2, 2, 2), direct=True
    )
    # First call projects the sheet corners; subsequent calls are actual rows.
    np.testing.assert_allclose(calls[1][0], [0.0, 0.5])
    np.testing.assert_allclose(calls[1][1], [1.0, 1.0])
    np.testing.assert_allclose(calls[2][0], [0.0, 0.5])
    np.testing.assert_allclose(calls[2][1], [0.5, 0.5])


def test_generic_latlon_points_and_feet_are_accumulated(tmp_path, monkeypatch):
    path = tmp_path / "terrain.gen"
    path.write_text("3 2 0\n1.25 2.25 100\n1.75 2.75 200\n9 9 -40000\n", encoding="ascii")
    monkeypatch.setattr(
        "pycalmet._terrel.project_geographic",
        lambda longitude, latitude, grid, datum: (longitude * 1000.0, latitude * 1000.0),
    )
    grid = GridConfig(2, 2, 0.5, 0.5, x_origin_km=2.0, y_origin_km=1.0)
    sums = np.zeros(4)
    counts = np.zeros(4, dtype=np.int64)
    inside = _accumulate_generic(path, grid, sums, counts, "WGS-84")
    assert inside == 2
    np.testing.assert_array_equal(counts, [1, 0, 0, 1])
    np.testing.assert_allclose(sums, [30.48, 0.0, 0.0, 60.96])


def test_common_accumulator_tracks_peak_independently_of_mean():
    grid = GridConfig(1, 1, 1.0, 1.0)
    sums = np.zeros(1)
    counts = np.zeros(1, dtype=np.int64)
    maxima = np.full(1, -999.0)
    assert _accumulate_points([100, 200, 300], [100, 200, 300],
                              [5, 20, 10], grid, sums, counts, maxima) == 3
    np.testing.assert_allclose(sums, [35.0])
    np.testing.assert_array_equal(counts, [3])
    np.testing.assert_allclose(maxima, [20.0])


def test_cartesian_peak_radius_duplicates_boundary_sample_into_neighbor():
    grid = GridConfig(2, 1, 1.0, 1.0)
    sums = np.zeros(2)
    counts = np.zeros(2, dtype=np.int64)
    maxima = np.full(2, -999.0)
    inside = _accumulate_points(
        [200.0, 950.0],
        [500.0, 500.0],
        [10.0, 5.0],
        grid,
        sums,
        counts,
        maxima,
        peak_radius_grid=0.1,
    )
    assert inside == 2
    np.testing.assert_allclose(sums, [15.0, 5.0])
    np.testing.assert_array_equal(counts, [2, 1])
    np.testing.assert_allclose(maxima, [10.0, 5.0])


def test_cartesian_peak_does_not_add_cross_cell_when_other_perturbation_is_outside():
    grid=GridConfig(2,2,1.,1.)
    sums=np.zeros(4,dtype=np.float32)
    counts=np.zeros(4,dtype=np.int64)
    maxima=np.full(4,-999.,dtype=np.float32)
    _accumulate_points(
        [1950.],[950.],[17.],grid,sums,counts,maxima,peak_radius_grid=.1)
    np.testing.assert_array_equal(counts,[0,1,0,0])


def test_polar_peak_does_not_add_angular_cell_when_radial_perturbation_is_outside():
    grid=GridConfig(
        2,4,1.,1.,grid_type=3,ring_distances_km=(1.,2.),
        ray_angles_deg=(10.,100.,190.,280.),polar_processing=1,
    )
    direction=np.float32(54.)
    distance=np.float32(2450.)
    radians=np.deg2rad(np.float64(direction))
    x=np.float32(distance*np.sin(radians)); y=np.float32(distance*np.cos(radians))
    sums=np.zeros(8,dtype=np.float32)
    counts=np.zeros(8,dtype=np.int64)
    maxima=np.full(8,-999.,dtype=np.float32)
    _accumulate_points(
        [x],[y],[23.],grid,sums,counts,maxima,peak_radius_grid=np.float32(100.))
    np.testing.assert_array_equal(counts,[0,1,0,0,0,0,0,0])


def test_generic_utm_kilometres_are_transformed_before_binning(tmp_path, monkeypatch):
    path = tmp_path / "terrain.gen"
    path.write_text("2 1 50\n500 3000 10\n501 3001 20\n", encoding="ascii")
    seen = []

    def fake_transform(x, y, source_grid, target_grid):
        seen.append((x.copy(), y.copy(), source_grid))
        return x, y

    monkeypatch.setattr("pycalmet._terrel.transform_grid", fake_transform)
    grid = GridConfig(2, 2, 1.0, 1.0, x_origin_km=500.0, y_origin_km=3000.0)
    sums = np.zeros(4)
    counts = np.zeros(4, dtype=np.int64)
    assert _accumulate_generic(path, grid, sums, counts, "NAS-C") == 2
    np.testing.assert_array_equal(seen[0][0], [500000.0, 501000.0])
    assert seen[0][2].utm_zone == 50 and seen[0][2].datum == "NAS-C"
    np.testing.assert_allclose(sums, [10.0, 0.0, 0.0, 20.0])


@pytest.mark.parametrize("header", ["0 1 50", "3 9 0", "1 1 61"])
def test_generic_header_validation(tmp_path, header):
    path = tmp_path / "bad.gen"
    path.write_text(f"{header}\n1 2 3\n", encoding="ascii")
    with pytest.raises(ValueError):
        _read_generic(path)


def test_nzgen_uses_id_elevation_longitude_latitude_order(tmp_path, monkeypatch):
    path = tmp_path / "new_zealand.dat"
    path.write_text("101 12 172.25 -41.75\n102 24 172.75 -41.25\n", encoding="ascii")
    seen = []

    def fake_project(longitude, latitude, grid, datum):
        seen.append((longitude.copy(), latitude.copy(), datum))
        return (longitude - 172.0) * 1000.0, (latitude + 42.0) * 1000.0

    monkeypatch.setattr("pycalmet._terrel.project_geographic", fake_project)
    grid = GridConfig(2, 2, 0.5, 0.5)
    sums = np.zeros(4)
    counts = np.zeros(4, dtype=np.int64)
    assert _accumulate_nzgen(path, grid, sums, counts, "WGS-84") == 2
    np.testing.assert_array_equal(seen[0][0], [172.25, 172.75])
    np.testing.assert_array_equal(seen[0][1], [-41.75, -41.25])
    np.testing.assert_array_equal(counts, [1, 0, 0, 1])
    np.testing.assert_allclose(sums, [12.0, 0.0, 0.0, 24.0])


def test_nzgen_requires_exactly_four_columns(tmp_path):
    path = tmp_path / "bad_nz.dat"
    path.write_text("1 2 3\n", encoding="ascii")
    with pytest.raises(ValueError, match="four-column"):
        _accumulate_nzgen(
            path,
            GridConfig(1, 1, 1.0, 1.0),
            np.zeros(1),
            np.zeros(1, dtype=np.int64),
        )


@pytest.mark.parametrize(
    ("name", "shape"),
    [("NADEM.IM", (9223, 8996)), ("EADEM_E.IM", (13000, 13000)),
     ("EADEM_A.IM", (13000, 12000))],
)
def test_usgsla_filename_selects_continent_geometry(name, shape):
    spec = _usgsla_spec(Path(name))
    assert (spec.nx, spec.ny) == shape


def test_usgsla_preserves_bottom_up_rows_and_legacy_base_255_decode(tmp_path, monkeypatch):
    path = tmp_path / "NADEM.IM"
    raw = np.array(
        [[[1, 2], [0, 3]], [[0, 4], [2, 1]]],
        dtype=np.uint8,
    )
    raw.tofile(path)
    spec = UsgslaSpec(2, 2, 0.0, 0.0, 0.0, 0.0)
    monkeypatch.setattr(
        "pycalmet._terrel.unproject_to_crs",
        lambda x, y, grid, crs, datum, units: (np.array([1.0, 2.0]), np.array([1.0, 2.0])),
    )
    monkeypatch.setattr(
        "pycalmet._terrel.project_glaz_to_lcc",
        lambda x, y, source_grid, target_grid, units: (x * 1000.0, y * 1000.0),
    )
    grid = GridConfig(2, 2, 1.0, 1.0, projection="LCC",
                      x_origin_km=1.0, y_origin_km=1.0)
    sums = np.zeros(4)
    counts = np.zeros(4, dtype=np.int64)
    assert _accumulate_usgsla(path, grid, sums, counts, spec=spec) == 4
    np.testing.assert_array_equal(counts, [1, 1, 1, 1])
    np.testing.assert_allclose(sums, [4.0, 511.0, 257.0, 3.0])


def test_arm3_reads_fixed_width_panel_and_legacy_sheet_coordinates(tmp_path, monkeypatch):
    path = tmp_path / "arm3.dat"
    elevations = np.full(120 * 120, 10, dtype=np.int64)
    elevations[:2] = [-1, -2]
    lines = ["".join(f"{value:10d}" for value in (1, 0, 0, 1, 1)),
             "".join(f"{value:20d}" for value in (1, 0))]
    lines.extend(
        "".join(f"{value:4d}" for value in elevations[start:start + 30])
        for start in range(0, elevations.size, 30)
    )
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    longitude, latitude, nx, ny, index, loaded = _read_arm3(path)
    assert (longitude, latitude, nx, ny) == (0, 0, 1, 1)
    np.testing.assert_array_equal(index, [1, 0])
    np.testing.assert_array_equal(loaded[:2], [-1, -2])
    monkeypatch.setattr(
        "pycalmet._terrel.project_geographic",
        lambda lon, lat, grid, datum: (lon * 1000.0, lat * 1000.0),
    )
    grid = GridConfig(3, 2, 0.5, 0.5, x_origin_km=-1.0)
    sums = np.zeros(6)
    counts = np.zeros(6, dtype=np.int64)
    assert _accumulate_arm3(path, grid, sums, counts) == 120 * 120
    assert counts.sum() == 120 * 120
    assert sums.sum() == 120 * 120 * 10 - 23


def test_3cd_filename_encodes_southeast_corner_in_west_longitude():
    assert _3cd_origin(Path("rockies40105.3CD")) == (40, 105)
    with pytest.raises(ValueError, match="not a 3CD"):
        _3cd_origin(Path("terrain.bin"))


def test_3cd_reads_longitude_strips_south_to_north(tmp_path, monkeypatch):
    path = tmp_path / "tile00000.3cd"
    np.arange(1, 10, dtype="<i2").reshape(3, 3).tofile(path)
    monkeypatch.setattr(
        "pycalmet._terrel.project_geographic",
        lambda lon, lat, grid, datum: (lon * 1000.0, lat * 1000.0),
    )
    grid = GridConfig(3, 3, 0.5, 0.5, x_origin_km=-1.0)
    sums = np.zeros(9)
    counts = np.zeros(9, dtype=np.int64)
    assert _accumulate_3cd(path, grid, sums, counts, samples=3) == 9
    np.testing.assert_array_equal(counts, np.ones(9, dtype=np.int64))
    np.testing.assert_array_equal(sums.reshape(3, 3), [[1, 4, 7], [2, 5, 8], [3, 6, 9]])


def _dmdf_record(kind, first=None, second=None):
    record = [" "] * 80
    record[:2] = kind
    for point, starts, widths in (
        (first, (14, 18, 25, 29, 36), (4, 6, 4, 6, 8)),
        (second, (45, 49, 56, 60, 67), (4, 6, 4, 6, 8)),
    ):
        if point is None:
            continue
        for value, start, width in zip(point, starts, widths):
            record[start:start + width] = f"{value:{width}d}"
    return "".join(record)


def test_dmdf_record_types_continuations_and_zone12_offset(tmp_path, monkeypatch):
    path = tmp_path / "canada.dmdf"
    lines = [
        _dmdf_record("07"),
        _dmdf_record("01", (500, 0, 3000, 0, 10000)),
        _dmdf_record("02", (501, 0, 3000, 0, 20000),
                     (1500, 0, 3000, 0, 30000)),
        _dmdf_record("06"),
        _dmdf_record("00", (999, 0, 999, 0, 99000)),
        _dmdf_record("03", (502, 0, 3000, 0, 40000),
                     (503, 0, 3000, 0, 50000)),
        _dmdf_record("00", (504, 0, 3000, 0, 60000)),
        _dmdf_record("99"),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")
    points = _read_dmdf(path)
    assert points.shape == (6, 3)
    np.testing.assert_allclose(points[:, 2], [10, 20, 30, 40, 50, 60])
    seen = []

    def fake_transform(x, y, source_grid, target_grid):
        seen.append((source_grid.utm_zone, x.copy()))
        return np.full(x.shape, 500500.0), np.full(y.shape, 3000500.0)

    monkeypatch.setattr("pycalmet._terrel.transform_grid", fake_transform)
    grid = GridConfig(1, 1, 1.0, 1.0, projection="UTM", utm_zone=11,
                      x_origin_km=500.0, y_origin_km=3000.0)
    sums = np.zeros(1)
    counts = np.zeros(1, dtype=np.int64)
    assert _accumulate_dmdf(path, grid, sums, counts) == 6
    assert [zone for zone, _ in seen] == [11, 12]
    np.testing.assert_array_equal(seen[1][1], [500000.0])
    np.testing.assert_allclose(sums, [210.0])
    np.testing.assert_array_equal(counts, [6])


def _put(record, start, width, value, *, floating=False):
    text = f"{value:{width}.15E}".replace("E", "D") if floating else f"{value:{width}d}"
    record[start:start + width] = text


def _dem_a_record():
    record = [" "] * 1024
    for index, value in enumerate((3, 0, 0, 0)):
        _put(record, 144 + 6 * index, 6, value)
    for index, value in enumerate((3, 2, 4)):
        _put(record, 528 + 6 * index, 6, value)
    corners = (-3600.0, 0.0, -3600.0, 3600.0, 0.0, 3600.0, 0.0, 0.0)
    for index, value in enumerate(corners):
        _put(record, 546 + 24 * index, 24, value, floating=True)
    for index, value in enumerate((1800.0, 1800.0, 1.0)):
        record[816 + 12 * index:828 + 12 * index] = f"{value:12.6E}".replace("E", "D")
    _put(record, 852, 6, 1)
    _put(record, 858, 6, 3)
    record[890:892] = " 3"
    return "".join(record)


def _dem_b_record(column, values, base=10.0, origin=(0.0, 0.0)):
    record = [" "] * 1024
    for index, value in enumerate((1, column, len(values), 1)):
        _put(record, 6 * index, 6, value)
    _put(record, 24, 24, origin[0], floating=True)
    _put(record, 48, 24, origin[1], floating=True)
    _put(record, 72, 24, base, floating=True)
    _put(record, 96, 24, min(values), floating=True)
    _put(record, 120, 24, max(values), floating=True)
    for index, value in enumerate(values):
        _put(record, 144 + 6 * index, 6, value)
    return "".join(record)


def test_usgs_degree_dem_parses_blocked_records_and_profiles(tmp_path, monkeypatch):
    path = tmp_path / "degree.dem"
    records = [_dem_a_record(), _dem_b_record(1, [1, 2, 3]),
               _dem_b_record(2, [4, 5, 6]), _dem_b_record(3, [7, 8, -32767])]
    path.write_bytes("".join(records).encode("ascii"))
    assert len(_dem_records(path)) == 4
    header = _parse_dem_header(records[0])
    assert header.codes == (3, 0, 0, 0)
    assert header.resolution == (1800.0, 1800.0, 1.0)
    assert header.profile_count == (1, 3) and header.horizontal_datum == 3
    seen = []

    def fake_project(lon, lat, grid, datum):
        seen.append(datum)
        return lon * 1000.0, lat * 1000.0

    monkeypatch.setattr("pycalmet._terrel.project_geographic", fake_project)
    grid = GridConfig(3, 3, 0.5, 0.5, x_origin_km=-1.0)
    sums = np.zeros(9)
    counts = np.zeros(9, dtype=np.int64)
    assert _accumulate_usgs_degree_dem(path, grid, sums, counts) == 8
    assert seen == ["WGS-84"]
    np.testing.assert_array_equal(counts.reshape(3, 3), [[1, 1, 1], [1, 1, 1], [1, 1, 0]])
    np.testing.assert_allclose(sums.reshape(3, 3), [[11, 14, 17], [12, 15, 18], [13, 16, 0]])


def test_usgs30_uses_utm_profile_origins_spacing_and_vertical_scale(tmp_path, monkeypatch):
    header = list(_dem_a_record())
    for index, value in enumerate((1, 0, 1, 50)):
        _put(header, 144 + 6 * index, 6, value)
    for index, value in enumerate((2, 2, 4)):
        _put(header, 528 + 6 * index, 6, value)
    for index, value in enumerate((30.0, 30.0, 2.0)):
        header[816 + 12 * index:828 + 12 * index] = f"{value:12.6E}".replace("E", "D")
    _put(header, 858, 6, 2)
    records = ["".join(header),
               _dem_b_record(1, [1, 2, 3], origin=(500015.0, 3000015.0)),
               _dem_b_record(2, [4, 5, 6], origin=(500045.0, 3000015.0))]
    path = tmp_path / "quad.dem"
    path.write_bytes("".join(records).encode("ascii"))
    seen = []

    def fake_transform(x, y, source_grid, target_grid):
        seen.append((source_grid.utm_zone, x.copy(), y.copy()))
        return x, y

    monkeypatch.setattr("pycalmet._terrel.transform_grid", fake_transform)
    grid = GridConfig(2, 3, 0.03, 0.03, projection="UTM", utm_zone=50,
                      x_origin_km=500.0, y_origin_km=3000.0)
    sums = np.zeros(6)
    counts = np.zeros(6, dtype=np.int64)
    assert _accumulate_usgs30(path, grid, sums, counts) == 6
    assert [entry[0] for entry in seen] == [50, 50]
    np.testing.assert_array_equal(seen[0][2], [3000015.0, 3000045.0, 3000075.0])
    np.testing.assert_array_equal(counts, np.ones(6, dtype=np.int64))
    np.testing.assert_allclose(sums.reshape(3, 2), [[22, 28], [24, 30], [26, 32]])


def test_usgs_dem_profile_continuation_records_hold_170_values():
    first = list(_dem_b_record(1, list(range(146))))
    _put(first, 12, 6, 317)
    continuation1 = [" "] * 1024
    continuation2 = [" "] * 1024
    for index, value in enumerate(range(146, 316)):
        _put(continuation1, 6 * index, 6, value)
    _put(continuation2, 0, 6, 316)
    profiles = _parse_dem_profiles(
        [_dem_a_record(), "".join(first), "".join(continuation1), "".join(continuation2)],
        1,
    )
    np.testing.assert_array_equal(profiles[0][3], np.arange(317))


def test_interp2d_reproduces_a_plane_inside_rectangle():
    value = _interp2d(
        [0, 2, 2, 0],
        [0, 0, 2, 2],
        [0, 2, 22, 20],
        np.sqrt([2, 2, 2, 2]),
        1.0,
        1.0,
        5.0,
    )
    assert value == pytest.approx(11.0)


def test_cellfill_uses_nearest_valid_cell_for_outer_boundary():
    grid = GridConfig(3, 3, 1.0, 1.0)
    sums = np.zeros(9)
    counts = np.zeros(9, dtype=np.int64)
    sums[4] = 10.0
    counts[4] = 1
    maxima = np.full(9, -999.0)
    maxima[4] = 20.0
    assert _fill_void_cells(sums, counts, grid, 2.0, maxima) == 8
    np.testing.assert_allclose(sums, 10.0)
    np.testing.assert_allclose(maxima, 20.0)
    np.testing.assert_array_equal(counts, 1)




def test_run_dispatches_generic_source_from_public_configuration(tmp_path, monkeypatch):
    source = tmp_path / "terrain.gen"
    source.write_text("3 1 0\n0 0 1\n", encoding="ascii")
    seen = []

    def fake_accumulate(path, grid, sums, counts, datum, *, maxima=None, peak=False,
                        receptors=None, terrain_filter=None):
        seen.append((path, datum))
        sums[0] = 12.0
        counts[0] = 1
        maxima[0] = 12.0
        return 1

    monkeypatch.setattr("pycalmet._terrel._accumulate_generic", fake_accumulate)
    output = tmp_path / "terrain.nc"
    config = TerrelConfig(
        tmp_path,
        output,
        GridConfig(1, 1, 1.0, 1.0),
        terrain_sources=[TerrelSource(source, "GEN", "NAS-C")],
    )
    result = run(config)
    assert seen == [(source, "NAS-C")]
    assert result.tiles == 1 and result.samples_inside_grid == 1


def test_polar_boundaries_and_cellind_radial_perturbation():
    grid = GridConfig(
        2,
        4,
        1.0,
        1.0,
        grid_type=3,
        ring_distances_km=(1.0, 2.0),
        ray_angles_deg=(10.0, 100.0, 190.0, 280.0),
        polar_processing=1,
    )
    radial, angular = _polar_boundaries(grid)
    np.testing.assert_allclose(radial, [1500.0, 2500.0])
    np.testing.assert_allclose(angular, [45.0, 135.0, 225.0, 315.0])
    angle = np.radians(10.0)
    x, y = 1000.0 * np.sin(angle), 1000.0 * np.cos(angle)
    sums = np.zeros(8)
    counts = np.zeros(8, dtype=np.int64)
    maxima = np.full(8, -999.0)
    assert _accumulate_points([x], [y], [12.0], grid, sums, counts, maxima, 600.0) == 1
    np.testing.assert_array_equal(counts[:2], [1, 1])
    np.testing.assert_allclose(maxima[:2], [12.0, 12.0])


def test_fillang_rejects_an_entirely_empty_ring_instead_of_looping_forever():
    with pytest.raises(ValueError,match="entirely empty polar ring"):
        _fill_polar_angular_gaps(np.full((4,1),-32000.,dtype=np.float32))


def test_polar_finalize_runs_fillang_after_height_replacement(tmp_path):
    grid=GridConfig(
        2,4,1.,1.,grid_type=3,ring_distances_km=(1.,2.),
        ray_angles_deg=(10.,100.,190.,280.),polar_processing=1,
    )
    config=TerrelConfig(tmp_path,tmp_path/"polar.nc",grid)
    terrain=_finalize_terrain(
        np.array([37.,51.,0.,0.,0.,0.,0.,0.],dtype=np.float32),
        np.array([1,1,0,0,0,0,0,0],dtype=np.int64),
        np.array([37.,51.,-999.,-999.,-999.,-999.,-999.,-999.],dtype=np.float32),
        grid,False,_terrain_filter(config),
    )
    np.testing.assert_array_equal(
        terrain,np.broadcast_to(np.array([37.,51.],dtype=np.float32),(4,2)))


def test_isc_cartesian_writer_emits_discrete_receptors_at_cell_centres(tmp_path):
    output = tmp_path / "isc.dat"
    grid = GridConfig(2, 1, 1.0, 1.0, x_origin_km=10.0, y_origin_km=20.0)
    _write_legacy(output, np.array([[12.0, 34.0]]), grid, "WGS-84", model=4)
    lines = output.read_text(encoding="ascii").splitlines()
    assert lines[0].startswith("TERREL.DAT")
    assert lines[-2:] == [
        "RE DISCCART   10500.0   20500.0    12.0",
        "RE DISCCART   11500.0   20500.0    34.0",
    ]


def test_isc_polar_writer_emits_gridpolr_records(tmp_path):
    output = tmp_path / "polar.dat"
    grid = GridConfig(2, 2, 1.0, 1.0, x_origin_km=10.0, y_origin_km=20.0,
                      grid_type=3, ring_distances_km=(1.0, 2.0),
                      ray_angles_deg=(10.0, 190.0), polar_processing=1)
    _write_legacy(output, np.array([[1.0, 2.0], [3.0, 4.0]]), grid, "WGS-84", model=3)
    lines = output.read_text(encoding="ascii").splitlines()
    start = lines.index("RE GRIDPOLR  XXXXX  STA")
    assert lines[0].startswith("TERREL.DAT")
    assert any("       7.500      17.500" in line for line in lines[:start])
    assert lines[start + 1] == "                    ORIG        10000.0        20000.0"
    assert lines[start + 2].endswith("  1000.0  2000.0")
    assert lines[start + 3].endswith("    10   190")
    assert lines[start + 4].endswith("    10     1.0     2.0")
    assert lines[-1] == "RE GRIDPOLR  XXXXX  END"


@pytest.mark.parametrize(
    ("model", "expected_rows"),
    [
        (1, ["     3.0     4.0", "     1.0     2.0"]),
        (2, ["    1    2", "    3    4"]),
        (5, ["     1.0     2.0", "     3.0     4.0"]),
    ],
)
def test_model_writer_uses_modout_row_order_and_format(tmp_path, model, expected_rows):
    output = tmp_path / f"model-{model}.dat"
    grid = GridConfig(2, 2, 1.0, 1.0, projection="UTM", utm_zone=50)
    _write_legacy(output, np.array([[1.0, 2.0], [3.0, 4.0]]),
                  grid, "WGS-84", model=model)
    assert output.read_text(encoding="ascii").splitlines()[-2:] == expected_rows


def test_generic_writer_adds_cell_centre_extent_record(tmp_path):
    output = tmp_path / "generic.dat"
    grid = GridConfig(2, 2, 1.0, 1.0, projection="UTM", utm_zone=50,
                      x_origin_km=10.0, y_origin_km=20.0)
    _write_legacy(output, np.array([[1.0, 2.0], [3.0, 4.0]]),
                  grid, "WGS-84", model=6)
    lines = output.read_text(encoding="ascii").splitlines()
    assert lines[-3] == " 2 2 10.5 20.5 11.5 21.5 1"
    assert lines[-2:] == ["    1    2", "    3    4"]


def test_cartesian_receptors_from_polar_grid_use_ring_geometry(tmp_path):
    output = tmp_path / "polar-receptors.dat"
    grid = GridConfig(2, 2, 1.0, 1.0, x_origin_km=10.0, y_origin_km=20.0,
                      grid_type=3, ring_distances_km=(1.0, 2.0),
                      ray_angles_deg=(90.0, 180.0))
    _write_legacy(output, np.array([[1.0, 2.0], [3.0, 4.0]]), grid, "WGS-84", model=4)
    assert output.read_text(encoding="ascii").splitlines()[-4:] == [
        "RE DISCCART   11000.0   20000.0     1.0",
        "RE DISCCART   12000.0   20000.0     2.0",
        "RE DISCCART   10000.0   19000.0     3.0",
        "RE DISCCART   10000.0   18000.0     4.0",
    ]


def test_public_isc_cartesian_run_outputs_peak_not_mean(tmp_path, monkeypatch):
    source = tmp_path / "peak.gen"
    source.write_text("1 1 50\n200 200 5\n300 300 20\n", encoding="ascii")
    monkeypatch.setattr(
        "pycalmet._terrel.transform_grid",
        lambda x, y, source_grid, target_grid: (x, y),
    )
    output = tmp_path / "peak.nc"
    config = TerrelConfig(
        tmp_path,
        output,
        GridConfig(1, 1, 1.0, 1.0, projection="UTM", utm_zone=50),
        model=4,
        terrain_sources=[TerrelSource(source, "GEN", "WGS-84")],
    )
    result = run(config)
    assert result.maximum_m == 20.0
    with Dataset(output) as dataset:
        assert dataset["terrain_height"].long_name.startswith("grid-cell peak")
        assert float(dataset["terrain_height"][0, 0]) == 20.0


def test_public_run_writes_peak_discrete_receptor_elevations(tmp_path, monkeypatch):
    source = tmp_path / "terrain.gen"
    source.write_text("1 1 50\n100 100 5\n200 100 20\n900 900 30\n", encoding="ascii")
    receptors = tmp_path / "receptors.inp"
    receptors.write_text("0.15 0.10\n0.90 0.90\n", encoding="ascii")
    monkeypatch.setattr(
        "pycalmet._terrel.transform_grid",
        lambda x, y, source_grid, target_grid: (x, y),
    )
    receptor_output = tmp_path / "receptors.out"
    config = TerrelConfig(
        tmp_path,
        tmp_path / "terrain.nc",
        GridConfig(2, 2, 1.0, 1.0, projection="UTM", utm_zone=50),
        terrain_sources=[TerrelSource(source, "GEN", "WGS-84")],
        receptor_input=receptors,
        receptor_output=receptor_output,
        receptor_columns=2,
        receptor_radius_km=0.1,
    )
    run(config)
    records = np.loadtxt(receptor_output)
    np.testing.assert_allclose(records[:, 2], [20.0, 30.0])
    np.testing.assert_allclose(records[:, 3], 0.0)


def test_public_run_interpolates_discrete_receptor_from_four_quadrants(
    tmp_path, monkeypatch
):
    source = tmp_path / "terrain.gen"
    source.write_text(
        "1 1 50\n0 0 0\n1000 0 10\n1000 1000 20\n0 1000 10\n",
        encoding="ascii",
    )
    receptors = tmp_path / "receptors.inp"
    receptors.write_text("0.5 0.5 99 2\n", encoding="ascii")
    monkeypatch.setattr(
        "pycalmet._terrel.transform_grid",
        lambda x, y, source_grid, target_grid: (x, y),
    )
    receptor_output = tmp_path / "receptors.out"
    config = TerrelConfig(
        tmp_path,
        tmp_path / "terrain.nc",
        GridConfig(2, 2, 1.0, 1.0, projection="UTM", utm_zone=50),
        terrain_sources=[TerrelSource(source, "GEN", "WGS-84")],
        receptor_input=receptors,
        receptor_output=receptor_output,
        receptor_columns=4,
        interpolate_receptors=True,
        receptor_radius_km=1.0,
    )
    run(config)
    record = np.loadtxt(receptor_output)
    np.testing.assert_allclose(record, [0.5, 0.5, 10.0, 2.0])


def test_continuation_save_matches_single_combined_run(tmp_path, monkeypatch):
    first = tmp_path / "first.gen"
    second = tmp_path / "second.gen"
    combined = tmp_path / "combined.gen"
    first.write_text("1 1 50\n100 100 10\n", encoding="ascii")
    second.write_text("1 1 50\n200 100 30\n", encoding="ascii")
    combined.write_text("1 1 50\n100 100 10\n200 100 30\n", encoding="ascii")
    monkeypatch.setattr(
        "pycalmet._terrel.transform_grid",
        lambda x, y, source_grid, target_grid: (x, y),
    )
    grid = GridConfig(1, 1, 1.0, 1.0, projection="UTM", utm_zone=50)
    save = tmp_path / "terrain.sav"
    run(TerrelConfig(
        tmp_path,
        tmp_path / "first.nc",
        grid,
        terrain_sources=[TerrelSource(first, "GEN", "WGS-84")],
        save_output=save,
    ))
    endian, records = _fortran_records(save)
    assert endian == "<"
    assert [len(record) for record in records] == [28, 4, 4, 4]
    run(TerrelConfig(
        tmp_path,
        tmp_path / "continued.nc",
        grid,
        terrain_sources=[TerrelSource(second, "GEN", "WGS-84")],
        previous_save=save,
    ))
    run(TerrelConfig(
        tmp_path,
        tmp_path / "combined.nc",
        grid,
        terrain_sources=[TerrelSource(combined, "GEN", "WGS-84")],
    ))
    with Dataset(tmp_path / "continued.nc") as continued, Dataset(
        tmp_path / "combined.nc"
    ) as expected:
        np.testing.assert_array_equal(
            continued["sample_count"][:], expected["sample_count"][:]
        )
        np.testing.assert_allclose(
            continued["terrain_height"][:], expected["terrain_height"][:]
        )


def test_continuation_rejects_mismatched_grid_header(tmp_path):
    save = tmp_path / "terrain.sav"
    grid = GridConfig(1, 1, 1.0, 1.0)
    _write_save(save, grid, np.array([1.0]), np.array([1]), np.array([1.0]), None)
    with pytest.raises(ValueError, match="does not match"):
        _load_save(
            save,
            GridConfig(2, 1, 1.0, 1.0),
            np.zeros(2),
            np.zeros(2, dtype=np.int64),
            np.zeros(2),
            None,
        )


def test_continuation_preserves_discrete_interpolation_vertices(tmp_path, monkeypatch):
    lower = tmp_path / "lower.gen"
    upper = tmp_path / "upper.gen"
    lower.write_text("1 1 50\n0 0 0\n1000 0 10\n", encoding="ascii")
    upper.write_text("1 1 50\n1000 1000 20\n0 1000 10\n", encoding="ascii")
    receptor_input = tmp_path / "receptors.inp"
    receptor_input.write_text("0.5 0.5\n", encoding="ascii")
    monkeypatch.setattr(
        "pycalmet._terrel.transform_grid",
        lambda x, y, source_grid, target_grid: (x, y),
    )
    grid = GridConfig(2, 2, 1.0, 1.0, projection="UTM", utm_zone=50)
    save = tmp_path / "terrain.sav"
    common = dict(
        srtm_dir=tmp_path,
        grid=grid,
        receptor_input=receptor_input,
        receptor_columns=2,
        interpolate_receptors=True,
        receptor_radius_km=1.0,
    )
    run(TerrelConfig(
        output=tmp_path / "lower.nc",
        terrain_sources=[TerrelSource(lower, "GEN", "WGS-84")],
        receptor_output=tmp_path / "lower-receptors.out",
        save_output=save,
        **common,
    ))
    output = tmp_path / "continued-receptors.out"
    run(TerrelConfig(
        output=tmp_path / "upper.nc",
        terrain_sources=[TerrelSource(upper, "GEN", "WGS-84")],
        receptor_output=output,
        previous_save=save,
        **common,
    ))
    np.testing.assert_allclose(np.loadtxt(output), [0.5, 0.5, 10.0, 0.0])


def test_polar_save_header_uses_outer_boundary_southwest_corner():
    grid = GridConfig(
        2,
        2,
        1.0,
        1.0,
        x_origin_km=10.0,
        y_origin_km=20.0,
        grid_type=3,
        ring_distances_km=(1.0, 2.0),
        ray_angles_deg=(10.0, 190.0),
        polar_processing=1,
    )
    # Normal IPROC boundary beyond the final 2 km ring is 2.5 km.
    np.testing.assert_array_equal(
        _save_header(grid, 0), [7500, 17500, 50, 1000, 2, 2, 0]
    )


def test_land_noise_and_missing_rules_apply_during_extraction(tmp_path, monkeypatch):
    source = tmp_path / "terrain.gen"
    source.write_text(
        "1 1 50\n100 100 -5\n200 100 -32768\n300 100 10\n",
        encoding="ascii",
    )
    monkeypatch.setattr(
        "pycalmet._terrel.transform_grid",
        lambda x, y, source_grid, target_grid: (x, y),
    )
    output = tmp_path / "terrain.nc"
    run(TerrelConfig(
        tmp_path,
        output,
        GridConfig(1, 1, 1.0, 1.0, projection="UTM", utm_zone=50),
        terrain_sources=[TerrelSource(source, "GEN", "WGS-84")],
        noise_replacement=(0, 2, 0, 0, 0),
        noise_minimum_m=(0.0, 1.0, 1.0, 1.0, 1.0),
        missing_replacement=(3, 2, 0, 0, 0),
        default_elevation_m=(0.0, 7.0, 0.0, 0.0, 0.0),
    ))
    with Dataset(output) as dataset:
        # -5 -> ZNOISE=1, void -> TERDEF=7, and 10 remains unchanged.
        assert float(dataset["terrain_height"][0, 0]) == 6.0
        assert int(dataset["sample_count"][0, 0]) == 3


def test_coastline_type_controls_always_default_replacement(tmp_path, monkeypatch):
    source = tmp_path / "terrain.gen"
    source.write_text("1 1 50\n250 500 10\n1250 500 20\n", encoding="ascii")
    coast = tmp_path / "coast.bln"
    coast.write_text(
        '5,0,"1,1,1"\n1.0,0.0\n2.0,0.0\n2.0,1.0\n1.0,1.0\n1.0,0.0\n',
        encoding="ascii",
    )
    monkeypatch.setattr(
        "pycalmet._terrel.transform_grid",
        lambda x, y, source_grid, target_grid: (x, y),
    )
    output = tmp_path / "terrain.nc"
    run(TerrelConfig(
        tmp_path,
        output,
        GridConfig(2, 1, 1.0, 1.0, projection="UTM", utm_zone=50),
        terrain_sources=[TerrelSource(source, "GEN", "WGS-84")],
        coastline_file=coast,
        missing_replacement=(3, 0, 0, 0, 0),
        default_elevation_m=(0.0, 0.0, 0.0, 0.0, 0.0),
    ))
    with Dataset(output) as dataset:
        np.testing.assert_allclose(dataset["terrain_height"][:], [[0.0, 20.0]])


def test_discrete_receptor_ignores_source_points_outside_grid(tmp_path, monkeypatch):
    source = tmp_path / "terrain.gen"
    source.write_text("1 1 50\n100 100 5\n1100 100 99\n", encoding="ascii")
    receptors = tmp_path / "receptors.inp"
    receptors.write_text("1.1 0.1\n", encoding="ascii")
    monkeypatch.setattr(
        "pycalmet._terrel.transform_grid",
        lambda x, y, source_grid, target_grid: (x, y),
    )
    receptor_output = tmp_path / "receptors.out"
    run(TerrelConfig(
        tmp_path,
        tmp_path / "terrain.nc",
        GridConfig(1, 1, 1.0, 1.0, projection="UTM", utm_zone=50),
        terrain_sources=[TerrelSource(source, "GEN", "WGS-84")],
        receptor_input=receptors,
        receptor_output=receptor_output,
        receptor_columns=2,
        receptor_radius_km=0.05,
    ))
    assert np.loadtxt(receptor_output)[2] == -999.0


def test_polar_coastline_uses_outer_ring_bounding_box(tmp_path):
    coast = tmp_path / "coast.bln"
    coast.write_text(
        '5,0,"1,1,1"\n7.5,17.5\n8.0,17.5\n8.0,18.0\n7.5,18.0\n7.5,17.5\n',
        encoding="ascii",
    )
    grid = GridConfig(
        2,
        2,
        1.0,
        1.0,
        x_origin_km=10.0,
        y_origin_km=20.0,
        grid_type=3,
        ring_distances_km=(1.0, 2.0),
        ray_angles_deg=(10.0, 190.0),
        polar_processing=1,
    )
    terrain_filter = _terrain_filter(TerrelConfig(tmp_path, tmp_path / "out.nc", grid,
                                                  coastline_file=coast))
    assert terrain_filter.coastal and len(terrain_filter.polygons) == 1


def test_surfer_plot_writer_uses_cell_centres_and_south_to_north_rows(tmp_path):
    output=tmp_path / "terrain.grd"
    grid=GridConfig(2,2,1.0,1.0,x_origin_km=10.0,y_origin_km=20.0)
    _write_surfer_grid(output,np.array([[1.0,2.0],[3.0,4.0]]),grid)
    assert output.read_text(encoding="ascii").splitlines()==[
        "DSAA",
        " 2 2",
        " 10.5 11.5",
        " 20.5 21.5",
        " 1 4",
        "     1.0     2.0",
        "     3.0     4.0",
    ]


def test_public_run_writes_configured_surfer_plot(tmp_path,monkeypatch):
    source=tmp_path / "terrain.gen"
    source.write_text("1 1 50\n100 100 5\n",encoding="ascii")
    monkeypatch.setattr("pycalmet._terrel.transform_grid",
                        lambda x,y,source_grid,target_grid:(x,y))
    plot=tmp_path / "qa.grd"
    run(TerrelConfig(
        tmp_path,tmp_path / "terrain.nc",
        GridConfig(1,1,1.0,1.0,projection="UTM",utm_zone=50),
        terrain_sources=[TerrelSource(source,"GEN","WGS-84")],
        plot_output=plot,
    ))
    assert plot.read_text(encoding="ascii").splitlines()[-1]=="     5.0"


def test_raw_echo_streams_inside_points_before_replacement(tmp_path,monkeypatch):
    source=tmp_path / "terrain.gen"
    source.write_text("1 1 50\n100 100 -5\n1100 100 99\n",encoding="ascii")
    monkeypatch.setattr("pycalmet._terrel.transform_grid",
                        lambda x,y,source_grid,target_grid:(x,y))
    raw=tmp_path / "raw.dat"
    run(TerrelConfig(
        tmp_path,tmp_path / "terrain.nc",
        GridConfig(1,1,1.0,1.0,projection="UTM",utm_zone=50),
        terrain_sources=[TerrelSource(source,"GEN","WGS-84")],
        raw_echo_output=raw,
        noise_replacement=(0,2,0,0,0),
        noise_minimum_m=(0.0,1.0,1.0,1.0,1.0),
    ))
    lines=raw.read_text(encoding="ascii").splitlines()
    assert lines[0].startswith("RAWECHO.DAT     1.0")
    assert lines[-1]=="      0.100      0.100      -5.000"
    assert not any("99.000" in line for line in lines)


def test_polar_plot_output_is_created_empty(tmp_path):
    output=tmp_path / "polar.grd"
    grid=GridConfig(2,2,1.0,1.0,grid_type=3,
                    ring_distances_km=(1.0,2.0),ray_angles_deg=(10.0,190.0))
    _write_plot_output(output,np.ones((2,2)),grid)
    assert output.is_file() and output.stat().st_size==0


def test_public_isc_polar_run_writes_ring_and_ray_netcdf(tmp_path, monkeypatch):
    source = tmp_path / "polar.gen"
    source.write_text("1 1 50\n174 985 5\n347 1970 20\n", encoding="ascii")
    monkeypatch.setattr(
        "pycalmet._terrel.transform_grid",
        lambda x, y, source_grid, target_grid: (x, y),
    )
    grid = GridConfig(2, 2, 1.0, 1.0, grid_type=3,
                      ring_distances_km=(1.0, 2.0),
                      ray_angles_deg=(10.0, 190.0), polar_processing=1)
    output = tmp_path / "polar.nc"
    result = run(TerrelConfig(
        tmp_path,
        output,
        grid,
        model=3,
        terrain_sources=[TerrelSource(source, "GEN", "WGS-84")],
    ))
    assert result.maximum_m == 20.0
    with Dataset(output) as dataset:
        assert set(dataset.dimensions) == {"ring", "ray"}
        np.testing.assert_allclose(dataset["ring_distance"][:], [1.0, 2.0])
        np.testing.assert_allclose(dataset["ray_angle"][:], [10.0, 190.0])
        assert dataset["terrain_height"].dimensions == ("ray", "ring")


def test_geotiff_uses_pixel_centres_and_loadtiff_elevation_limits(tmp_path, monkeypatch):
    source = tmp_path / "terrain.tif"
    keys = (1, 1, 0, 4, 1024, 0, 1, 2, 1025, 0, 1, 1, 2048, 0, 1, 4326,
            4099, 0, 1, 9002)
    tifffile.imwrite(
        source,
        np.array([[10, 70000], [-40000, 40]], dtype=np.int32),
        extratags=[
            (33550, "d", 3, (0.5, 0.5, 0.0), False),
            (33922, "d", 6, (0.0, 0.0, 0.0, 0.0, 1.0, 0.0), False),
            (34735, "H", len(keys), keys, False),
        ],
    )
    crs, x0, y0, dx, dy, elevation_scale, data = _geotiff_geometry(source)
    assert crs.to_epsg() == 4326
    assert (x0, y0, dx, dy, data.shape) == (0.25, 0.75, 0.5, 0.5, (2, 2))
    assert elevation_scale == 0.3048
    monkeypatch.setattr(
        "pycalmet._terrel.project_geographic",
        lambda longitude, latitude, grid, datum: (longitude * 1000.0, latitude * 1000.0),
    )
    grid = GridConfig(2, 2, 0.5, 0.5)
    sums = np.zeros(4)
    counts = np.zeros(4, dtype=np.int64)
    assert _accumulate_geotiff(source, grid, sums, counts) == 4
    np.testing.assert_array_equal(counts, [1, 1, 1, 1])
    np.testing.assert_allclose(sums, [0.0, 12.192, 3.048, 0.0])


def test_geotiff_orientation_four_is_normalized_to_top_down_rows(tmp_path):
    source=tmp_path/'bottom-up.tif'
    keys=(1,1,0,3,1024,0,1,2,1025,0,1,1,2048,0,1,4326)
    tifffile.imwrite(
        source,np.array([[1,2],[3,4]],dtype=np.int16),
        extratags=[
            (274,'H',1,4,False),
            (33550,'d',3,(.5,.5,0.),False),
            (33922,'d',6,(0.,0.,0.,0.,0.,0.),False),
            (34735,'H',len(keys),keys,False),
        ],
    )
    _,x0,y0,dx,dy,_,data=_geotiff_geometry(source)
    assert (x0,y0,dx,dy)==(.25,.75,.5,.5)
    np.testing.assert_array_equal(data,[[3,4],[1,2]])
