#!/usr/bin/env python3
"""Check named K3 shard coverage and file length against safetensors headers."""
import argparse
import json
import struct
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path, default=Path('/hgpfs/Kimi-K3'))
    parser.add_argument('--index', type=Path, default=None)
    parser.add_argument('--output', type=Path, default=Path(__file__).parent / 'results/gb200/hgpfs_checkpoint_inventory.json')
    args = parser.parse_args()
    args.index = args.index or args.model_dir / 'model.safetensors.index.json'
    index = json.loads(args.index.read_text())
    required = set(index['weight_map'].values())
    files = []
    for path in sorted(args.model_dir.glob('*.safetensors')):
        with path.open('rb') as stream:
            length = struct.unpack('<Q', stream.read(8))[0]
            if length > 64 * 2**20:
                raise ValueError(f'Implausible safetensors header length: {path}')
            header = json.loads(stream.read(length))
        tensors = [v for k, v in header.items() if k != '__metadata__']
        expected = 8 + length + max(v['data_offsets'][1] for v in tensors)
        actual = path.stat().st_size
        files.append(dict(file=path.name, file_bytes=actual, expected_bytes=expected,
                          complete=actual >= expected, tensors=len(tensors)))
    result = dict(path=str(args.model_dir), index_path=str(args.index),
                  expected_shards=len(required), present_shards=len(files), files=files,
                  missing_shards=sorted(required - {f['file'] for f in files}),
                  index_total_bytes=index['metadata']['total_size'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(f"{len(files)}/{len(required)} shards present; {sum(f['complete'] for f in files)} cover their declared payload")


if __name__ == '__main__':
    main()
