"""GNU Fortran sequential-unformatted records used by CALMET.DAT."""
from __future__ import annotations
import struct
from pathlib import Path

class FortranSequentialWriter:
    """Write little-endian gfortran records with four-byte count markers.

    Attributes:
        path: Destination file path.
        stream: Open binary output stream, or ``None`` after closure.
        max_subrecord_bytes: Maximum payload bytes in one physical subrecord.
    """
    def __init__(self,path,*,max_subrecord_bytes=2_147_483_639):
        """Open a sequential-unformatted output file.

        Args:
            path: Destination path, truncated if it already exists.
            max_subrecord_bytes: Positive int32-sized physical payload limit;
                smaller values are useful for testing chained subrecords.

        Raises:
            ValueError: If ``max_subrecord_bytes`` cannot be stored in a
                positive signed 32-bit record marker.
            OSError: If the destination cannot be opened.
        """
        if not 1<=max_subrecord_bytes<=2_147_483_647:
            raise ValueError('max_subrecord_bytes must fit a positive int32 marker')
        self.path=Path(path); self.stream=self.path.open('wb')
        self.max_subrecord_bytes=int(max_subrecord_bytes)
    def write_record(self,*parts):
        """Write one logical record from bytes-like payload parts.

        Args:
            *parts: Bytes-like objects whose raw bytes are concatenated into
                one logical Fortran record.

        Raises:
            OSError: If the record cannot be written.
            TypeError: If a part does not expose a byte-compatible buffer.
        """
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
        """Close the output stream; repeated calls have no effect."""
        if self.stream is not None: self.stream.close(); self.stream=None
    def __enter__(self):
        """Enter a context-manager block.

        Returns:
            This open writer.
        """
        return self
    def __exit__(self,*exc):
        """Close the writer when leaving a context-manager block.

        Args:
            *exc: Exception information supplied by the context protocol.
        """
        self.close()

class FortranSequentialReader:
    """Read logical gfortran records, including chained subrecords.

    Attributes:
        stream: Open binary input stream, or ``None`` after closure.
    """
    def __init__(self,path):
        """Open a sequential-unformatted input file.

        Args:
            path: Existing file path.

        Raises:
            OSError: If the source cannot be opened.
        """
        self.stream=Path(path).open('rb')
    def read_record(self):
        """Read and join the next logical Fortran record.

        Returns:
            Record payload bytes, or ``None`` at a clean end of file.

        Raises:
            ValueError: If a marker, payload, or chained-subrecord sign is
                truncated or inconsistent.
            OSError: If the stream cannot be read.
        """
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
        """Close the input stream; repeated calls have no effect."""
        if self.stream is not None: self.stream.close(); self.stream=None
    def __enter__(self):
        """Enter a context-manager block.

        Returns:
            This open reader.
        """
        return self
    def __exit__(self,*exc):
        """Close the reader when leaving a context-manager block.

        Args:
            *exc: Exception information supplied by the context protocol.
        """
        self.close()

def fixed_string(value,length):
    """Encode a blank-padded Fortran ASCII character value.

    Args:
        value: Value converted to text before encoding.
        length: Exact output width in bytes.

    Returns:
        ASCII bytes padded on the right with spaces to ``length``.

    Raises:
        UnicodeEncodeError: If the text contains non-ASCII characters.
        ValueError: If the encoded value exceeds ``length``.
    """
    encoded=str(value).encode('ascii')
    if len(encoded)>length: raise ValueError(f'string exceeds Fortran CHARACTER*{length}')
    return encoded.ljust(length,b' ')
