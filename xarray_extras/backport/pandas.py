import os
import pandas


def to_csv(x, path_or_buf=None, header=True, encoding='utf-8',
           line_terminator=os.linesep, **kwargs):
    """Compatibility layer for Series.to_csv() and DataFrame.to_csv()
    for pandas < 0.24:

    - add support for line_terminator parameter; if omitted it defaults to
      os.linesep
    - header parameter defaults to True also for Series; suppress warning
      when omitted.
    """
    if pandas.__version__ >= '0.24':
        return x.to_csv(path_or_buf, header=header, encoding=encoding,
                        line_terminator=line_terminator, **kwargs)

    out = x.to_csv(header=header, **kwargs)
    if line_terminator != '\n':
        out = out.replace('\n', line_terminator)

    if path_or_buf is None:
        return out

    if isinstance(path_or_buf, str):
        with open(path_or_buf, 'w', encoding=encoding) as fh:
            fh.write(out)
    else:
        path_or_buf.write(out)
    return None
