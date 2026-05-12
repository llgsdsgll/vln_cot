import os
from io import StringIO

import numpy as np


_DTYPE_MAP = {
    ('F', 4): np.dtype('<f4'),
    ('F', 8): np.dtype('<f8'),
    ('I', 1): np.dtype('<i1'),
    ('I', 2): np.dtype('<i2'),
    ('I', 4): np.dtype('<i4'),
    ('I', 8): np.dtype('<i8'),
    ('U', 1): np.dtype('<u1'),
    ('U', 2): np.dtype('<u2'),
    ('U', 4): np.dtype('<u4'),
    ('U', 8): np.dtype('<u8'),
}


def _normalize_points(points):
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f'Expected points with shape (N, 3), got {pts.shape}')
    return pts


def _normalize_colors(colors, n_points):
    if colors is None:
        return None
    cols = np.asarray(colors, dtype=np.float32)
    if cols.ndim != 2 or cols.shape != (n_points, 3):
        raise ValueError(f'Expected colors with shape ({n_points}, 3), got {cols.shape}')
    return np.clip(cols, 0.0, 1.0)


def _pack_rgb_float(colors):
    cols = np.rint(colors * 255.0).astype(np.uint8)
    packed = (
        (cols[:, 0].astype(np.uint32) << 16)
        | (cols[:, 1].astype(np.uint32) << 8)
        | cols[:, 2].astype(np.uint32)
    ).astype('<u4')
    return packed.view('<f4')


def _unpack_rgb_float(rgb_field):
    packed = np.asarray(rgb_field, dtype='<f4').view('<u4')
    colors = np.empty((len(packed), 3), dtype=np.float32)
    colors[:, 0] = ((packed >> 16) & 0xFF).astype(np.float32) / 255.0
    colors[:, 1] = ((packed >> 8) & 0xFF).astype(np.float32) / 255.0
    colors[:, 2] = (packed & 0xFF).astype(np.float32) / 255.0
    return colors


def write_pcd(path, points, colors=None):
    pts = _normalize_points(points)
    cols = _normalize_colors(colors, len(pts))

    if cols is None:
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4')])
        fields = 'x y z'
        sizes = '4 4 4'
        types = 'F F F'
        counts = '1 1 1'
    else:
        dtype = np.dtype([('x', '<f4'), ('y', '<f4'), ('z', '<f4'), ('rgb', '<f4')])
        fields = 'x y z rgb'
        sizes = '4 4 4 4'
        types = 'F F F F'
        counts = '1 1 1 1'

    cloud = np.empty(len(pts), dtype=dtype)
    cloud['x'] = pts[:, 0]
    cloud['y'] = pts[:, 1]
    cloud['z'] = pts[:, 2]
    if cols is not None:
        cloud['rgb'] = _pack_rgb_float(cols)

    header = '\n'.join([
        '# .PCD v0.7 - Point Cloud Data file format',
        'VERSION 0.7',
        f'FIELDS {fields}',
        f'SIZE {sizes}',
        f'TYPE {types}',
        f'COUNT {counts}',
        f'WIDTH {len(pts)}',
        'HEIGHT 1',
        'VIEWPOINT 0 0 0 1 0 0 0',
        f'POINTS {len(pts)}',
        'DATA binary',
        '',
    ])

    output_dir = os.path.dirname(path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        f.write(cloud.tobytes())


def _build_structured_dtype(fields, sizes, types, counts):
    dtype_fields = []
    expanded_names = []
    for field, size, field_type, count in zip(fields, sizes, types, counts):
        if (field_type, size) not in _DTYPE_MAP:
            raise ValueError(f'Unsupported PCD field type {field_type}{size} for field {field}')
        base_dtype = _DTYPE_MAP[(field_type, size)]
        if count == 1:
            dtype_fields.append((field, base_dtype))
            expanded_names.append(field)
        else:
            dtype_fields.append((field, base_dtype, (count,)))
            expanded_names.extend([field] * count)
    return np.dtype(dtype_fields), expanded_names


def _parse_header(f):
    header = {}
    while True:
        raw_line = f.readline()
        if not raw_line:
            raise ValueError('Unexpected end of PCD header')
        line = raw_line.decode('ascii', errors='strict').strip()
        if not line or line.startswith('#'):
            continue
        key, *values = line.split()
        header[key.upper()] = values
        if key.upper() == 'DATA':
            return header


def _header_value(header, key, default=None):
    values = header.get(key)
    if values is None:
        return default
    if len(values) == 1:
        return values[0]
    return values


def _header_list(header, key, default=None):
    values = header.get(key)
    if values is None:
        return default
    return list(values)


def read_pcd(path):
    with open(path, 'rb') as f:
        header = _parse_header(f)

        fields = _header_list(header, 'FIELDS', [])
        sizes = [int(v) for v in _header_list(header, 'SIZE', [])]
        types = _header_list(header, 'TYPE', [])
        counts = _header_list(header, 'COUNT', None)
        counts = [int(v) for v in counts] if counts is not None else [1] * len(fields)
        points = int(_header_value(header, 'POINTS', _header_value(header, 'WIDTH', 0)))
        data_type = _header_value(header, 'DATA', '').lower()

        if not (len(fields) == len(sizes) == len(types) == len(counts)):
            raise ValueError(f'Inconsistent PCD header in {path}')

        dtype, expanded_names = _build_structured_dtype(fields, sizes, types, counts)

        if data_type == 'binary':
            payload = f.read()
            cloud = np.frombuffer(payload, dtype=dtype, count=points)
            if len(cloud) != points:
                raise ValueError(f'Expected {points} points in {path}, got {len(cloud)}')
        elif data_type == 'ascii':
            if any(count != 1 for count in counts):
                raise ValueError('ASCII PCD with COUNT > 1 is not supported')
            payload = f.read().decode('ascii', errors='strict').strip()
            if payload:
                matrix = np.loadtxt(StringIO(payload), dtype=np.float64)
                matrix = np.atleast_2d(matrix)
            else:
                matrix = np.zeros((0, len(expanded_names)), dtype=np.float64)
            if matrix.shape[1] != len(expanded_names):
                raise ValueError(f'Unexpected ASCII PCD column count in {path}')
            cloud = np.empty(len(matrix), dtype=dtype)
            for col, name in enumerate(expanded_names):
                cloud[name] = matrix[:, col].astype(cloud.dtype[name])
        else:
            raise ValueError(f'Unsupported PCD DATA mode: {data_type}')

    required = {'x', 'y', 'z'}
    if not required.issubset(cloud.dtype.names):
        raise ValueError(f'PCD file {path} is missing one of {sorted(required)}')

    points_xyz = np.stack([cloud['x'], cloud['y'], cloud['z']], axis=1).astype(np.float32, copy=False)

    colors = None
    if 'rgb' in cloud.dtype.names:
        colors = _unpack_rgb_float(cloud['rgb'])
    elif 'rgba' in cloud.dtype.names:
        colors = _unpack_rgb_float(cloud['rgba'])

    return points_xyz, colors
