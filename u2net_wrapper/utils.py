"""Utility functions."""

import hashlib
import requests
from pathlib import Path
from clint.textui import progress

def download_file(url, save_filepath, chunk_size=8196):
    """
    Downloads a file from the specified URL.
    :param filepath:
        The path to the file.
    :param chunk_size:
        An integer representing the number of bytes of data to get
        in a single iteration. Defaults to 8196.
    """

    request = requests.get(url, stream=True)
    with open(save_filepath, 'wb+') as file:
        total_length = int(request.headers.get('content-length'))
        expected_size = (total_length / chunk_size) + 1
        for chunk in progress.bar(request.iter_content(chunk_size=chunk_size),
                                  expected_size=expected_size,
                                  label=f'{save_filepath.name} '):
            file.write(chunk)
            file.flush()

def get_md5_from_file(filepath, chunk_size=8196):
    """
    Gets the MD5 hash of a file.
    :param filepath:
        The path to the file.
    :param chunk_size:
        An integer representing the number of bytes of data to get
        in a single iteration. Defaults to 8196.
    :returns:
        The MD5 hash of the file, or None if it doesn't exist.
    """

    filepath = Path(filepath)
    if not filepath.exists(): return

    h = hashlib.md5()
    with open(filepath, 'rb') as file:
        chunk = 0
        while chunk != b'':
            chunk = file.read(chunk_size)
            h.update(chunk)

    return h.hexdigest()