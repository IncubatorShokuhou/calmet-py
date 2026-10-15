"""GNU Fortran sequential-unformatted records used by CALMET.DAT."""
from __future__ import annotations
import struct
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, cast

class FortranSequentialWriter:
    """Write little-endian gfortran records with four-byte count markers.

    Attributes:
        path: Destination file path.
        stream: Open binary output stream, or ``None`` after closure.
        max_subrecord_bytes: Maximum payload bytes in one physical subrecord.
    """
    def __init__(
        self,
        path: str | Path,
        *,
        max_subrecord_bytes: int = 2_147_483_639,
    ) -> None:
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
        self.path: Path=Path(path); self.stream: BinaryIO | None=self.path.open('wb')
        self.max_subrecord_bytes: int=int(max_subrecord_bytes)
    def write_record(self,*parts: Any) -> None:
        """Write one logical record from bytes-like payload parts.

        Args:
            *parts: Bytes-like objects whose raw bytes are concatenated into
                one logical Fortran record.

        Raises:
            OSError: If the record cannot be written.
            TypeError: If a part does not expose a byte-compatible buffer.
        """
        payload: bytes=b''.join(memoryview(part).tobytes() if not isinstance(part,(bytes,bytearray)) else bytes(part) for part in parts)
        if not payload:
            marker: bytes=struct.pack('<i',0)
            cast(BinaryIO,self.stream).write(marker); cast(BinaryIO,self.stream).write(marker)
            return
        view: memoryview=memoryview(payload)
        offsets: range=range(0,len(payload),self.max_subrecord_bytes)
        for index,offset in enumerate(offsets):
            chunk: memoryview=view[offset:offset+self.max_subrecord_bytes]
            size: int=len(chunk)
            leading: int=-size if offset+size<len(payload) else size
            trailing: int=-size if index else size
            cast(BinaryIO,self.stream).write(struct.pack('<i',leading))
            cast(BinaryIO,self.stream).write(chunk)
            cast(BinaryIO,self.stream).write(struct.pack('<i',trailing))
    def close(self) -> None:
        """Close the output stream; repeated calls have no effect."""
        if self.stream is not None: self.stream.close(); self.stream=None
    def __enter__(self) -> FortranSequentialWriter:
        """Enter a context-manager block.

        Returns:
            This open writer.
        """
        return self
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the writer when leaving a context-manager block.

        Args:
            exc_type: Exception class supplied by the context manager protocol.
            exc_value: Exception instance supplied by the context manager
                protocol.
            traceback: Exception traceback supplied by the context manager
                protocol.
        """
        self.close()

class FortranSequentialReader:
    """Read logical gfortran records, including chained subrecords.

    Attributes:
        stream: Open binary input stream, or ``None`` after closure.
    """
    def __init__(self,path: str | Path) -> None:
        """Open a sequential-unformatted input file.

        Args:
            path: Existing file path.

        Raises:
            OSError: If the source cannot be opened.
        """
        self.stream: BinaryIO | None=Path(path).open('rb')
    def read_record(self) -> bytes | None:
        """Read and join the next logical Fortran record.

        Returns:
            Record payload bytes, or ``None`` at a clean end of file.

        Raises:
            ValueError: If a marker, payload, or chained-subrecord sign is
                truncated or inconsistent.
            OSError: If the stream cannot be read.
        """
        marker: bytes=cast(BinaryIO,self.stream).read(4)
        if not marker: return None
        if len(marker)!=4: raise ValueError('truncated leading Fortran record marker')
        parts: list[bytes]=[]; first: bool=True
        while True:
            leading: int=struct.unpack('<i',marker)[0]
            if leading==-2_147_483_648:
                raise ValueError('invalid Fortran subrecord size')
            size: int=abs(leading)
            payload: bytes=cast(BinaryIO,self.stream).read(size); trailing_marker: bytes=cast(BinaryIO,self.stream).read(4)
            if len(payload)!=size or len(trailing_marker)!=4:
                raise ValueError('invalid or truncated Fortran record')
            trailing: int=struct.unpack('<i',trailing_marker)[0]
            if abs(trailing)!=size or (trailing<0)==first:
                raise ValueError('invalid Fortran subrecord markers')
            parts.append(payload)
            if leading>=0: return b''.join(parts)
            marker=cast(BinaryIO,self.stream).read(4)
            if len(marker)!=4:
                raise ValueError('truncated leading Fortran subrecord marker')
            first=False
    def close(self) -> None:
        """Close the input stream; repeated calls have no effect."""
        if self.stream is not None: self.stream.close(); self.stream=None
    def __enter__(self) -> FortranSequentialReader:
        """Enter a context-manager block.

        Returns:
            This open reader.
        """
        return self
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Close the reader when leaving a context-manager block.

        Args:
            exc_type: Exception class supplied by the context manager protocol.
            exc_value: Exception instance supplied by the context manager
                protocol.
            traceback: Exception traceback supplied by the context manager
                protocol.
        """
        self.close()

def fixed_string(value: object,length: int) -> bytes:
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
    encoded: bytes=str(value).encode('ascii')
    if len(encoded)>length: raise ValueError(f'string exceeds Fortran CHARACTER*{length}')
    return encoded.ljust(length,b' ')
