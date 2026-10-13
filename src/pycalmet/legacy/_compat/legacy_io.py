"""GNU Fortran sequential-unformatted records used by CALMET.DAT."""
from __future__ import annotations
import struct
from pathlib import Path

class FortranSequentialWriter:
    """Write little-endian gfortran records with four-byte byte-count markers."""
    def __init__(self,path,*,max_subrecord_bytes=2_147_483_639):
        if not 1<=max_subrecord_bytes<=2_147_483_647:
            raise ValueError('max_subrecord_bytes must fit a positive int32 marker')
        self.path=Path(path); self.stream=self.path.open('wb')
        self.max_subrecord_bytes=int(max_subrecord_bytes)
    def write_record(self,*parts):
        payload=b''.join(memoryview(part).tobytes() if not isinstance(part,(bytes,bytearray)) else bytes(part) for part in parts)
        if not payload:
            marker=struct.pack('<i',0)
            self.stream.write(marker); self.stream.write(marker)
            return
        view=memoryview(payload)
        offsets=range(0,len(payload),self.max_subrecord_bytes)
        for index,offset in enumerate(offsets):
            chunk=view[offset:offset+self.max_subrecord_bytes]
            size=len(chunk)
            leading=-size if offset+size<len(payload) else size
            trailing=-size if index else size
            self.stream.write(struct.pack('<i',leading))
            self.stream.write(chunk)
            self.stream.write(struct.pack('<i',trailing))
    def close(self):
        if self.stream is not None: self.stream.close(); self.stream=None
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()

class FortranSequentialReader:
    """Read logical gfortran records, including chained subrecords."""
    def __init__(self,path): self.stream=Path(path).open('rb')
    def read_record(self):
        marker=self.stream.read(4)
        if not marker: return None
        if len(marker)!=4: raise ValueError('truncated leading Fortran record marker')
        parts=[]; first=True
        while True:
            leading=struct.unpack('<i',marker)[0]
            if leading==-2_147_483_648:
                raise ValueError('invalid Fortran subrecord size')
            size=abs(leading)
            payload=self.stream.read(size); trailing_marker=self.stream.read(4)
            if len(payload)!=size or len(trailing_marker)!=4:
                raise ValueError('invalid or truncated Fortran record')
            trailing=struct.unpack('<i',trailing_marker)[0]
            if abs(trailing)!=size or (trailing<0)==first:
                raise ValueError('invalid Fortran subrecord markers')
            parts.append(payload)
            if leading>=0: return b''.join(parts)
            marker=self.stream.read(4)
            if len(marker)!=4:
                raise ValueError('truncated leading Fortran subrecord marker')
            first=False
    def close(self):
        if self.stream is not None: self.stream.close(); self.stream=None
    def __enter__(self): return self
    def __exit__(self,*exc): self.close()

def fixed_string(value,length):
    """Fortran blank-padded ASCII character value."""
    encoded=str(value).encode('ascii')
    if len(encoded)>length: raise ValueError(f'string exceeds Fortran CHARACTER*{length}')
    return encoded.ljust(length,b' ')
