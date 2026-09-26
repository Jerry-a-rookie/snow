#!/usr/bin/env python3
"""Convert PatchSTG NPZ traffic arrays to memory-mappable float32 NPY files."""

import argparse
import csv
import json
import os
import shutil
import zipfile
from pathlib import Path

import numpy as np


DATASETS = {
    "CA": ("flowca.npz", "CA/ca_meta.csv"),
    "GBA": ("flowgba.npz", "GBA/gba_meta.csv"),
    "GLA": ("flowgla.npz", "GLA/gla_meta.csv"),
    "SD": ("flowsd.npz", "SD/sd_meta.csv"),
}
CONVERTER_VERSION = 1


def _read_npy_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_1_0(stream)
    elif version == (2, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    elif version == (3, 0):
        shape, fortran_order, dtype = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NPY format version: {version}")
    return tuple(shape), bool(fortran_order), np.dtype(dtype)


def inspect_npz(npz_path, key="data"):
    """Return the embedded NPY member metadata without materializing its array."""
    npz_path = Path(npz_path)
    member_name = f"{key}.npy"
    with zipfile.ZipFile(npz_path) as archive:
        try:
            info = archive.getinfo(member_name)
        except KeyError as exc:
            raise KeyError(f"{npz_path} does not contain key {key!r}") from exc
        with archive.open(info, "r") as stream:
            shape, fortran_order, dtype = _read_npy_header(stream)
    return {
        "member": member_name,
        "shape": shape,
        "fortran_order": fortran_order,
        "dtype": dtype,
        "crc32": f"{info.CRC:08x}",
        "compressed_bytes": info.compress_size,
        "uncompressed_bytes": info.file_size,
    }


def _metadata_row_count(path):
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def _combine_stats(count, mean, m2, values):
    values = np.asarray(values, dtype=np.float32)
    chunk_count = values.size
    if chunk_count == 0:
        return count, mean, m2
    chunk_mean = float(np.mean(values, dtype=np.float64))
    centered = values.astype(np.float64, copy=False) - chunk_mean
    chunk_m2 = float(np.sum(centered * centered, dtype=np.float64))
    if count == 0:
        return chunk_count, chunk_mean, chunk_m2
    delta = chunk_mean - mean
    total = count + chunk_count
    combined_mean = mean + delta * chunk_count / total
    combined_m2 = m2 + chunk_m2 + delta * delta * count * chunk_count / total
    return total, combined_mean, combined_m2


def _expected_manifest(source_path, source_info, metadata_rows, train_ratio, val_ratio):
    raw_shape = source_info["shape"]
    if len(raw_shape) == 3 and raw_shape[-1] == 1:
        output_shape = raw_shape[:2]
    elif len(raw_shape) == 2:
        output_shape = raw_shape
    else:
        raise ValueError(
            f"Expected traffic data shaped [time, nodes] or [time, nodes, 1], got {raw_shape}"
        )
    if metadata_rows != output_shape[1]:
        raise ValueError(
            f"Metadata has {metadata_rows} rows but flow data has {output_shape[1]} nodes"
        )
    test_ratio = 1.0 - train_ratio - val_ratio
    num_train = int(output_shape[0] * train_ratio)
    num_test = int(output_shape[0] * test_ratio)
    val_end = output_shape[0] - num_test
    return {
        "converter_version": CONVERTER_VERSION,
        "source": {
            "path": str(Path(source_path).resolve()),
            "size_bytes": Path(source_path).stat().st_size,
            "member": source_info["member"],
            "member_crc32": source_info["crc32"],
            "shape": list(raw_shape),
            "dtype": str(source_info["dtype"]),
        },
        "flow_file": "flow.npy",
        "flow_shape": list(output_shape),
        "flow_dtype": "float32",
        "time_steps": output_shape[0],
        "nodes": output_shape[1],
        "metadata_file": "meta.csv",
        "metadata_rows": metadata_rows,
        "split": {
            "train_ratio": train_ratio,
            "val_ratio": val_ratio,
            "test_ratio": test_ratio,
            "train_end": num_train,
            "val_end": val_end,
        },
        "time_encoding": {"tod": 96, "dow": 7},
    }


def _is_current(output_dir, expected):
    manifest_path = output_dir / "manifest.json"
    flow_path = output_dir / "flow.npy"
    meta_path = output_dir / "meta.csv"
    if not (manifest_path.is_file() and flow_path.is_file() and meta_path.is_file()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        flow = np.load(flow_path, mmap_mode="r")
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    comparable_keys = (
        "converter_version",
        "source",
        "flow_file",
        "flow_shape",
        "flow_dtype",
        "time_steps",
        "nodes",
        "metadata_file",
        "metadata_rows",
        "split",
        "time_encoding",
    )
    return (
        all(manifest.get(key) == expected.get(key) for key in comparable_keys)
        and list(flow.shape) == expected["flow_shape"]
        and flow.dtype == np.dtype("float32")
        and "train_mean" in manifest
        and "train_std" in manifest
        and _metadata_row_count(meta_path) == expected["metadata_rows"]
    )


def convert_dataset(
    source_path,
    metadata_path,
    output_dir,
    *,
    key="data",
    train_ratio=0.6,
    val_ratio=0.2,
    chunk_bytes=64 * 1024 * 1024,
    overwrite=False,
):
    """Stream one NPZ member into a float32 NPY and return its manifest."""
    source_path = Path(source_path)
    metadata_path = Path(metadata_path)
    output_dir = Path(output_dir)
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise ValueError("Split ratios must satisfy train > 0, val >= 0, train + val < 1")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive")
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    source_info = inspect_npz(source_path, key=key)
    expected = _expected_manifest(
        source_path,
        source_info,
        _metadata_row_count(metadata_path),
        train_ratio,
        val_ratio,
    )
    if _is_current(output_dir, expected):
        return json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(
            f"{output_dir} contains an incomplete or incompatible conversion; use --overwrite"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    flow_path = output_dir / "flow.npy"
    temp_flow_path = output_dir / "flow.npy.tmp"
    temp_manifest_path = output_dir / "manifest.json.tmp"
    for temp_path in (temp_flow_path, temp_manifest_path):
        if temp_path.exists():
            temp_path.unlink()

    output_shape = tuple(expected["flow_shape"])
    source_dtype = source_info["dtype"]
    train_end = expected["split"]["train_end"]
    stat_count = 0
    stat_mean = 0.0
    stat_m2 = 0.0

    target = np.lib.format.open_memmap(
        temp_flow_path,
        mode="w+",
        dtype=np.float32,
        shape=output_shape,
    )
    try:
        with zipfile.ZipFile(source_path) as archive:
            with archive.open(source_info["member"], "r") as stream:
                shape, fortran_order, dtype = _read_npy_header(stream)
                if (
                    shape != source_info["shape"]
                    or fortran_order != source_info["fortran_order"]
                    or dtype != source_dtype
                ):
                    raise ValueError("Embedded NPY header changed while converting")

                if fortran_order:
                    values_per_column = int(
                        shape[0] * np.prod(shape[2:], dtype=np.int64)
                    )
                    columns_per_chunk = max(
                        1,
                        chunk_bytes // max(values_per_column * source_dtype.itemsize, 1),
                    )
                    for column_start in range(0, output_shape[1], columns_per_chunk):
                        column_end = min(
                            column_start + columns_per_chunk,
                            output_shape[1],
                        )
                        value_count = (
                            (column_end - column_start) * values_per_column
                        )
                        byte_count = value_count * source_dtype.itemsize
                        raw = stream.read(byte_count)
                        if len(raw) != byte_count:
                            raise EOFError(
                                f"Unexpected end of {source_info['member']}: "
                                f"wanted {byte_count} bytes, received {len(raw)}"
                            )
                        chunk = np.frombuffer(
                            raw,
                            dtype=source_dtype,
                            count=value_count,
                        ).reshape(
                            (shape[0], column_end - column_start) + shape[2:],
                            order="F",
                        )
                        if chunk.ndim == 3:
                            chunk = chunk[..., 0]
                        converted = chunk.astype(np.float32, copy=False)
                        target[:, column_start:column_end] = converted
                        stat_count, stat_mean, stat_m2 = _combine_stats(
                            stat_count,
                            stat_mean,
                            stat_m2,
                            converted[:train_end],
                        )
                else:
                    values_per_row = int(
                        np.prod(shape[1:], dtype=np.int64)
                    )
                    rows_per_chunk = max(
                        1,
                        chunk_bytes // max(values_per_row * source_dtype.itemsize, 1),
                    )
                    for row_start in range(0, output_shape[0], rows_per_chunk):
                        row_end = min(row_start + rows_per_chunk, output_shape[0])
                        value_count = (row_end - row_start) * values_per_row
                        byte_count = value_count * source_dtype.itemsize
                        raw = stream.read(byte_count)
                        if len(raw) != byte_count:
                            raise EOFError(
                                f"Unexpected end of {source_info['member']}: "
                                f"wanted {byte_count} bytes, received {len(raw)}"
                            )
                        chunk = np.frombuffer(
                            raw,
                            dtype=source_dtype,
                            count=value_count,
                        ).reshape((row_end - row_start,) + shape[1:])
                        if chunk.ndim == 3:
                            chunk = chunk[..., 0]
                        converted = chunk.astype(np.float32, copy=False)
                        target[row_start:row_end] = converted
                        stats_end = min(row_end, train_end)
                        if stats_end > row_start:
                            stat_count, stat_mean, stat_m2 = _combine_stats(
                                stat_count,
                                stat_mean,
                                stat_m2,
                                converted[:stats_end - row_start],
                            )
                if stream.read(1):
                    raise ValueError("Embedded NPY contains trailing array data")
        target.flush()
    except Exception:
        del target
        if temp_flow_path.exists():
            temp_flow_path.unlink()
        raise
    del target

    train_std = float(np.sqrt(stat_m2 / stat_count))
    if not np.isfinite(train_std) or train_std <= 0:
        if temp_flow_path.exists():
            temp_flow_path.unlink()
        raise ValueError(f"Training standard deviation must be positive, got {train_std}")
    manifest = dict(expected)
    manifest["train_mean"] = stat_mean
    manifest["train_std"] = train_std
    manifest["statistics"] = {
        "scope": "global_scalar_first_train_split",
        "count": stat_count,
    }

    shutil.copy2(metadata_path, output_dir / "meta.csv")
    temp_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_flow_path, flow_path)
    os.replace(temp_manifest_path, output_dir / "manifest.json")
    return manifest


def parse_args():
    script_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=script_root / "PatchSTG-main" / "data",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=script_root / "dataset" / "PatchSTG",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(DATASETS),
        default=list(DATASETS),
    )
    parser.add_argument("--chunk-mb", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    for dataset in args.datasets:
        flow_name, meta_name = DATASETS[dataset]
        manifest = convert_dataset(
            args.source_root / flow_name,
            args.source_root / meta_name,
            args.output_root / dataset,
            chunk_bytes=args.chunk_mb * 1024 * 1024,
            overwrite=args.overwrite,
        )
        print(
            f"{dataset}: shape={tuple(manifest['flow_shape'])}, "
            f"mean={manifest['train_mean']:.6f}, std={manifest['train_std']:.6f}"
        )


if __name__ == "__main__":
    main()
