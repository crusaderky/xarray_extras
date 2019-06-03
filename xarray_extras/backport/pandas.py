import os
from typing import Callable, Optional, Union
import pandas


def to_csv(x: Union[pandas.Series, pandas.DataFrame],
           path_or_buf=None, header=True, encoding='utf-8',
           line_terminator=os.linesep, compression='infer', **kwargs):
    """Compatibility layer for :meth:`pandas.Series.to_csv` and
    :meth:`pandas.DataFrame.to_csv`

    - add line_terminator parameter to pandas < 0.24;
      if omitted it defaults to os.linesep
    - header parameter defaults to True for Series in pandas < 0.24;
      suppress warning when omitted
    - add compression parameter to pandas < 0.23
    """
    if pandas.__version__ >= '0.24':
        return x.to_csv(path_or_buf, header=header, encoding=encoding,
                        line_terminator=line_terminator,
                        compression=compression, **kwargs)

    out = x.to_csv(header=header, **kwargs)
    if line_terminator != '\n':
        out = out.replace('\n', line_terminator)

    if path_or_buf is None:
        return out

    if not isinstance(path_or_buf, str):
        path_or_buf.write(out)
        return None

    open_func = _open_func(path_or_buf, compression)
    with open_func(path_or_buf, 'wt', encoding=encoding) as fh:
        fh.write(out)
    return None


def _open_func(path: str, compression: Optional[str]) -> Callable:
    """Find an open()-like function from the compression parameter.
    If compression='infer', figure it out from the path.
    """
    if compression == 'infer':
        compression = path.split('.')[-1].lower()
        if compression == 'gz':
            compression = 'gzip'
        elif compression == 'csv':
            compression = None

    if compression is None:
        return open
    if compression == 'gzip':
        import gzip
        return gzip.open
    if compression == 'bz2':
        import bz2
        return bz2.open
    if compression == 'xz':
        import lzma
        return lzma.open
    if compression == 'zip':
        raise NotImplementedError("zip compression is not supported")
    raise ValueError('Unknown compression: %s' % compression)
