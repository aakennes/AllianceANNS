"""Check relocatable source references without compiling or opening datasets.

This complements CMake/compiler checks: it catches orphaned search sources and
stale documentation links even when the affected code is not built on this OS.
Frozen experiment evidence and embedded dependencies are outside this scope.
"""
from pathlib import Path
import re
import tempfile
import unittest
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
CPP_SUFFIXES = {".h", ".hpp", ".cpp", ".cc", ".cxx"}
EMBEDDED = {"third", "rabitq"}


def active_sources(root):
    for directory in (root / "include", root / "src"):
        for path in directory.rglob("*"):
            if path.suffix in CPP_SUFFIXES and not EMBEDDED.intersection(path.relative_to(root).parts):
                yield path


def include_errors(root):
    errors = []
    if (root / "include/alliance").exists():
        errors.append("include/alliance: retired public include layer still exists")
    search_roots = [root / name for name in
                    ("include", "include/rabitq", "src/alliance", "src/tools", "src/test")]
    for source in active_sources(root):
        for line, text in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
            match = re.match(r'^\s*#\s*include\s*(["<])([^">]+)[">]', text)
            if not match:
                continue
            delimiter, name = match.groups()
            where = "{}:{}".format(source.relative_to(root), line)
            if name.startswith("alliance/"):
                errors.append("{}: retired include {}".format(where, name))
            elif delimiter == '"' and not any((directory / name).is_file() for directory in [source.parent] + search_roots):
                errors.append("{}: missing local include {}".format(where, name))
    return errors


def cmake_files(root):
    yield root / "CMakeLists.txt"
    yield from (root / "cmake").rglob("*.cmake")
    yield from (root / "src").rglob("CMakeLists.txt")
    yield from (root / "src").rglob("*.cmake")


def cmake_path(token, source, root):
    for name, value in (("PROJECT_SOURCE_DIR", root), ("CMAKE_SOURCE_DIR", root),
                        ("CMAKE_CURRENT_SOURCE_DIR", source.parent)):
        token = token.replace("${" + name + "}", str(value))
    # Loops, generator expressions and dependency paths are evaluated by CMake.
    if "$" in token:
        return None
    path = Path(token)
    return path if path.is_absolute() else source.parent / path


def cmake_errors(root):
    errors = []
    registered_search = set()
    for source in cmake_files(root):
        if not source.is_file():
            errors.append("missing CMake file: {}".format(source.relative_to(root)))
            continue
        text = re.sub(r"#[^\n]*", "", source.read_text(encoding="utf-8"))
        if "include/alliance" in text:
            errors.append("{}: retired public include directory".format(source.relative_to(root)))
        # Only literal file arguments are inspected; CMake remains responsible
        # for interpreting its language and resolving variable-generated lists.
        for quoted, bare in re.findall(r'"([^"]*)"|([^\s()"]+)', text):
            token = quoted or bare
            if Path(token).suffix not in {".cpp", ".cc", ".cxx", ".cmake", ".py"}:
                continue
            path = cmake_path(token, source, root)
            if path is not None and not path.is_file():
                errors.append("{}: missing CMake file reference {}".format(source.relative_to(root), token))
        for directory in re.findall(r"\badd_subdirectory\s*\(\s*([^\s)]+)", text):
            path = cmake_path(directory.strip('"'), source, root)
            if path is not None and not (path / "CMakeLists.txt").is_file():
                errors.append("{}: missing CMake subdirectory {}".format(source.relative_to(root), directory))
        for body in re.findall(r"\badd_library\s*\(\s*alliance_search_objects\s+OBJECT\s+([^)]*)\)", text):
            for token in body.split():
                path = cmake_path(token.strip('"'), source, root)
                if path is not None and path.suffix == ".cpp":
                    registered_search.add(path.resolve())
    for source in (root / "src/alliance/search").rglob("*.cpp"):
        if source.name.startswith("index_disk_"):
            errors.append("{}: retired search filename prefix".format(source.relative_to(root)))
        if source.resolve() not in registered_search:
            errors.append("{}: search source is not registered in alliance_search_objects".format(source.relative_to(root)))
    return errors


def active_documents(root):
    yield from root.glob("*.md")
    for folder in ("docs", "src", "tests"):
        for path in (root / folder).rglob("*.md"):
            relative = path.relative_to(root)
            if EMBEDDED.intersection(relative.parts):
                continue
            # Dated review/round reports describe frozen historical layouts.
            if folder == "docs" and re.match(r"(?:refactor_round|round\d+_|(?:pipeline|review)_cleanup_\d{8})", path.name):
                continue
            yield path
    yield from (root / "scripts").glob("README*.md")


def document_errors(root):
    errors = []
    for source in active_documents(root):
        text = source.read_text(encoding="utf-8")
        text = re.sub(r"(?ms)^\s*(```|~~~).*?^\s*\1[^\n]*$", "", text)
        links = re.findall(r"\]\(\s*(?:<([^>]+)>|([^\s)]+))", text)
        links += re.findall(r"(?m)^\s*\[[^\]]+\]:\s*(?:<([^>]+)>|(\S+))", text)
        for angle, plain in links:
            target = angle or plain
            url = urlsplit(target)
            if url.scheme or url.netloc or not url.path:
                continue
            name = unquote(url.path)
            path = root / name.lstrip("/") if name.startswith("/") else source.parent / name
            if not path.exists():
                errors.append("{}: missing local document link {}".format(source.relative_to(root), target))
    return errors


class RepositoryStructureTests(unittest.TestCase):
    def test_public_and_private_includes_resolve(self):
        self.assertEqual(include_errors(ROOT), [])

    def test_cmake_references_and_search_registration(self):
        self.assertEqual(cmake_errors(ROOT), [])

    def test_active_document_links_resolve(self):
        self.assertEqual(document_errors(ROOT), [])

    def test_faults_are_detected_without_a_compiler(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write(name, text):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                return path

            write("CMakeLists.txt", "add_subdirectory(src)\n")
            cmake = write("src/CMakeLists.txt", "add_library(alliance_search_objects OBJECT alliance/search/example.cpp)\n")
            source = write("src/alliance/search/example.cpp", '#include "search/api.hpp"\n')
            write("include/search/api.hpp", "#pragma once\n")
            document = write("README.md", "[API](include/search/api.hpp)\n")
            self.assertEqual(include_errors(root) + cmake_errors(root) + document_errors(root), [])

            for bad, reason in (("alliance/search/api.hpp", "retired include"),
                                ("search/missing.hpp", "missing local include")):
                with self.subTest(include=bad):
                    source.write_text('#include "{}"\n'.format(bad), encoding="utf-8")
                    self.assertTrue(any(reason in error for error in include_errors(root)))
            source.write_text('#include <alliance/search/api.hpp>\n', encoding="utf-8")
            self.assertTrue(any("retired include" in error for error in include_errors(root)))
            source.write_text('#include "search/api.hpp"\n', encoding="utf-8")
            cmake.write_text("add_library(alliance_search_objects OBJECT alliance/search/missing.cpp)\n", encoding="utf-8")
            errors = cmake_errors(root)
            self.assertTrue(any("missing CMake file reference" in error for error in errors))
            self.assertTrue(any("not registered" in error for error in errors))
            write("src/alliance/search/index_disk_example.cpp", "")
            self.assertTrue(any("retired search filename prefix" in error for error in cmake_errors(root)))
            document.write_text("[API](include/search/missing.hpp)\n", encoding="utf-8")
            self.assertTrue(any("missing local document link" in error for error in document_errors(root)))


if __name__ == "__main__":
    unittest.main()
