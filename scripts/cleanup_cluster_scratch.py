#!/usr/bin/env python3
"""Inspect one crashed cluster spill generation; remove only with explicit --apply.

This is disposable-stage cleanup, not a build/checkpoint recovery mechanism.
Never infer ownership from graph-name patterns, PID alone, or missing manifests.
"""
import argparse
import json
import os
from pathlib import Path
import re
import stat


def require(condition, message):
    if not condition:
        raise ValueError(message)


def same_file(entry, device, inode):
    return stat.S_ISREG(entry.st_mode) and entry.st_nlink == 1 and (
        entry.st_dev, entry.st_ino) == (device, inode)


def owner_is_active(manifest):
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot != manifest['boot_id']:
        return False
    try:
        process = Path('/proc/{}/stat'.format(manifest['pid'])).read_text()
    except FileNotFoundError:
        return False
    fields = process[process.rindex(')') + 2:].split()
    return fields[19] == manifest['start_ticks']


def inspect_or_remove(graph, generation, apply=False):
    graph, generation = Path(graph).absolute(), Path(generation).absolute()
    require(not graph.is_symlink() and not generation.is_symlink(), 'symlink path rejected')
    require(graph.parent.resolve() == generation.parent.resolve(), 'scratch must be beside graph')
    require(re.fullmatch(re.escape(graph.name) + r'\.cluster-scratch\.[A-Za-z0-9]{6}',
                         generation.name) is not None, 'not an exact scratch generation path')
    source = graph.stat()
    directory = generation.lstat()
    require(stat.S_ISREG(source.st_mode), 'source graph is not regular')
    require(stat.S_ISDIR(directory.st_mode) and directory.st_uid == os.getuid(),
            'scratch directory type or owner mismatch')
    require(directory.st_mode & 0o077 == 0, 'scratch directory must be private')
    directory_fd = os.open(generation, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        opened = os.fstat(directory_fd)
        require((opened.st_dev, opened.st_ino) == (directory.st_dev, directory.st_ino),
                'scratch directory changed')
        manifest_fd = os.open('manifest.json', os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory_fd)
        try:
            manifest_stat = os.fstat(manifest_fd)
            require(stat.S_ISREG(manifest_stat.st_mode) and manifest_stat.st_nlink == 1 and
                    manifest_stat.st_uid == os.getuid() and manifest_stat.st_size <= 16 << 20,
                    'invalid manifest file')
            with os.fdopen(manifest_fd, 'r', closefd=False) as stream:
                manifest = json.load(stream)
        finally:
            os.close(manifest_fd)
        require(manifest.get('version') == 1 and manifest.get('kind') == 'alliance-cluster-spill',
                'unknown manifest schema')
        require(Path(manifest['graph']) == graph and
                (manifest['graph_dev'], manifest['graph_ino']) == (source.st_dev, source.st_ino),
                'source graph identity mismatch')
        require(isinstance(manifest['pid'], int) and manifest['pid'] > 0 and
                isinstance(manifest['start_ticks'], str) and manifest['start_ticks'].isdigit() and
                isinstance(manifest['boot_id'], str) and
                re.fullmatch(r'[a-f0-9-]{36}', manifest['boot_id']), 'invalid owner identity')
        require(not owner_is_active(manifest), 'owner process is still active')
        names, present, logical, allocated = set(), set(), 0, 0
        for record in manifest['files']:
            name = record['name']
            require(isinstance(name, str) and (name == 'input.records' or re.fullmatch(r'cluster\.[0-9]+', name)) and
                    name not in names, 'invalid or duplicate owned filename')
            names.add(name)
            try:
                entry = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            except FileNotFoundError:
                require(record.get('allow_absent') is True, 'missing owned file: ' + name)
                continue
            present.add(name)
            require(same_file(entry, record['dev'], record['ino']) and entry.st_uid == os.getuid(),
                    'owned file identity changed: ' + name)
            logical += entry.st_size
            allocated += entry.st_blocks * 512
        require(set(os.listdir(directory_fd)) == present | {'manifest.json'},
                'unexpected or missing file; preserving generation')
        report = dict(generation=str(generation), action='remove' if apply else 'would_remove',
                      files=len(present), already_consumed=len(names - present), logical_bytes=logical, allocated_bytes=allocated,
                      owner_active=False)
        if apply:
            require(not owner_is_active(manifest), 'owner process became active')
            current_source = graph.stat()
            require((current_source.st_dev, current_source.st_ino) == (source.st_dev, source.st_ino),
                    'source graph changed before cleanup')
            for record in manifest['files']:
                if record['name'] not in present:
                    continue
                entry = os.stat(record['name'], dir_fd=directory_fd, follow_symlinks=False)
                require(same_file(entry, record['dev'], record['ino']), 'owned file changed before cleanup')
            current_manifest = os.stat('manifest.json', dir_fd=directory_fd, follow_symlinks=False)
            require(same_file(current_manifest, manifest_stat.st_dev, manifest_stat.st_ino),
                    'manifest changed before cleanup')
            for name in sorted(present):
                os.unlink(name, dir_fd=directory_fd)
            os.unlink('manifest.json', dir_fd=directory_fd)
            os.fsync(directory_fd)
            current_directory = generation.lstat()
            require((current_directory.st_dev, current_directory.st_ino) ==
                    (directory.st_dev, directory.st_ino), 'scratch directory path changed')
            os.rmdir(generation)
            parent_fd = os.open(generation.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(parent_fd)
            finally:
                os.close(parent_fd)
        return report
    finally:
        os.close(directory_fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', required=True)
    parser.add_argument('--generation', required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    print(json.dumps(inspect_or_remove(args.graph, args.generation, args.apply), sort_keys=True))


if __name__ == '__main__':
    main()
