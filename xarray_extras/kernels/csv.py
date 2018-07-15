import gzip
from multiprocessing import Process, Queue


def to_csv(x, constructor, indices, kwargs):
    x = constructor(x, *indices)
    queue = Queue()
    p = Process(target=to_csv_subprocess, args=(queue, x, kwargs))
    p.start()
    p.join()  # this blocks until the process terminates
    return queue.get()


def to_csv_subprocess(queue, x, kwargs):
    compression = kwargs.pop('compression', None)
    out = x.to_csv(**kwargs)
    out = out.encode('utf8')

    if compression == 'gzip':
        out = gzip.compress(out)
    elif compression is not None:
        raise NotImplementedError("Only gzip compression is supported")

    queue.put(out)


def to_file(fname, mode, data, rr_token=None):
    with open(fname, mode) as fh:
        fh.write(data)
