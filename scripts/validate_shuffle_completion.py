#!/usr/bin/env python3
"""Validate a committed shuffle layout before automatic workflow reuse."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def validate_completion(graph, layout, meta):
    """Read only the three expected outputs; reject incomplete or changed files."""
    graph = Path(graph)
    receipt = Path(str(graph) + '.completion.json')
    record = json.loads(receipt.read_text())
    if (not isinstance(record, dict) or record.get('status') != 'complete'
            or type(record.get('protocol_version')) is not int or record.get('protocol_version') != 1
            or record.get('ml_only') is not True):
        raise ValueError('shuffle completion is not a committed ML-only v1 receipt')
    expected = {Path(p).resolve(strict=True) for p in (graph, layout, meta)}
    entries = record.get('files')
    if len(expected) != 3 or not isinstance(entries, list) or len(entries) != 3:
        raise ValueError('shuffle completion must cover exactly three distinct outputs')
    checked = []
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
            raise ValueError('invalid shuffle completion file record')
        stored = Path(entry['path'])
        if not stored.is_absolute():
            raise ValueError('shuffle completion file paths must be absolute')
        actual = stored.resolve(strict=True)
        if actual not in expected or not actual.is_file():
            raise ValueError('shuffle completion path mismatch or duplicate')
        expected.remove(actual)
        size, sha = entry.get('bytes'), entry.get('sha256')
        if type(size) is not int or size < 0 or not isinstance(sha, str) or not re.fullmatch('[0-9a-f]{64}', sha):
            raise ValueError('invalid shuffle completion size or SHA-256')
        if actual.stat().st_size != size:
            raise ValueError('shuffle completion size mismatch: ' + str(actual))
        checked.append((actual, sha))
    for actual, sha in checked:
        before = actual.stat()
        if digest(actual) != sha:
            raise ValueError('shuffle completion SHA-256 mismatch: ' + str(actual))
        after = actual.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns):
            raise ValueError('shuffle output changed during validation: ' + str(actual))
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('graph')
    parser.add_argument('layout')
    parser.add_argument('meta')
    args = parser.parse_args()
    try:
        validate_completion(args.graph, args.layout, args.meta)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, 'invalid shuffle completion: ' + str(error) + '\n')


if __name__ == '__main__':
    main()
