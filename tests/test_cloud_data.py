from datetime import datetime
import struct
import numpy as np
import pytest

from pycalmet.cloud_data import CloudData,CloudDataWriter
from pycalmet.legacy_io import FortranSequentialWriter,fixed_string


def test_formatted_cloud_data_uses_interval_end_and_fortran_grid_order(tmp_path):
    path=tmp_path/'cloud.dat'
    path.write_text('"CLOUDFRA" 201800100 0 201800101 0  .1 .2 .3 .4 .5 .6\n',encoding='ascii')
    cloud=CloudData(path,3,2,2)
    np.testing.assert_allclose(cloud.at_end(datetime(2018,1,1,1)),[[.1,.2,.3],[.4,.5,.6]])
    with pytest.raises(ValueError,match='no record'):
        cloud.at_end(datetime(2018,1,1,2))


def test_unformatted_cloud_data_reads_gfortran_record(tmp_path):
    path=tmp_path/'cloud.bin'; values=np.arange(6,dtype='<f4')/10
    with FortranSequentialWriter(path) as writer:
        writer.write_record(fixed_string('CLOUDFRA',8),struct.pack('<4i',201800100,0,201800101,0),values)
    cloud=CloudData(path,3,2,1)
    np.testing.assert_allclose(cloud.at_end(datetime(2018,1,1,1)),values.reshape(2,3))


def test_cloud_data_rejects_fraction_outside_unit_interval(tmp_path):
    path=tmp_path/'cloud.dat'
    path.write_text('CLOUDFRA 201800100 0 201800101 0 1.1\n',encoding='ascii')
    with pytest.raises(ValueError,match='outside'):
        CloudData(path,1,1,2)


@pytest.mark.parametrize('format,suffix',[(1,'bin'),(2,'dat')])
def test_cloud_data_writer_roundtrips_outcld_records(tmp_path,format,suffix):
    path=tmp_path/f'cloud.{suffix}'
    begin=datetime(2018,1,1,0,15); end=datetime(2018,1,1,0,30)
    values=np.arange(6,dtype=np.float32).reshape(2,3)/10
    with CloudDataWriter(path,3,2,format) as writer:
        writer.append(begin,end,values)
    cloud=CloudData(path,3,2,format)
    np.testing.assert_array_equal(cloud.at_end(end),values)
