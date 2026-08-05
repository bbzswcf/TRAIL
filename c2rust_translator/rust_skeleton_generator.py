from __future__ import annotations

import re
import shutil
from pathlib import Path


BIN_MAIN_STUB_RE = re.compile(
    r"(?P<prefix>\b(?:pub\s+)?fn\s+main\s*\(\s*\)\s*\{\s*)"
    r"(?:unimplemented!|todo!)\s*\(\s*\)\s*;?"
    r"(?P<suffix>\s*\})",
    re.DOTALL,
)

ROOT_FIXTURE_SKIP_NAMES = {
    ".gitignore",
    "cargo.lock",
    "cargo.toml",
    "cmakelists.txt",
    "copying",
    "license",
    "makefile",
    "readme",
    "readme.md",
}
ROOT_FIXTURE_SKIP_SUFFIXES = {
    ".a",
    ".bib",
    ".bz2",
    ".c",
    ".cc",
    ".cmake",
    ".cpp",
    ".dylib",
    ".exe",
    ".gz",
    ".h",
    ".hpp",
    ".i",
    ".lock",
    ".log",
    ".md",
    ".o",
    ".pdf",
    ".rs",
    ".so",
    ".tar",
    ".tex",
    ".toml",
    ".xz",
    ".zip",
}


class Skeleton:
    def __init__(self, project_dir: str, output_dir: str, clean: bool = True):
        self.project_dir = Path(project_dir)
        self.output_dir = Path(output_dir)
        self.clean = clean
        self.module_files: dict[str, str] = {}

    def generate(self) -> str:
        if not self.project_dir.exists():
            raise FileNotFoundError(f"RBench path not found: {self.project_dir}")

        if self.clean and self.output_dir.exists():
            if self.output_dir.resolve() == self.project_dir.resolve():
                raise RuntimeError("Refuse to clean output dir because it equals source dir")
            shutil.rmtree(self.output_dir)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._copy("Cargo.toml")
        self._copy("Cargo.lock")
        self._copy_src()
        self._copy_root_fixtures()
        self._index_modules()
        return str(self.output_dir)

    def _copy(self, name: str) -> None:
        src = self.project_dir / name
        if not src.is_file():
            return
        dst = self.output_dir / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def _copy_src(self) -> None:
        src = self.project_dir / "src"
        if src.is_dir():
            shutil.copytree(src, self.output_dir / "src", dirs_exist_ok=True)
            self._neutralize_unimplemented_bin_mains()

    def _copy_root_fixtures(self) -> None:
        source_project = self._matching_cbench_project_dir()
        if source_project is None:
            return
        for path in sorted(source_project.iterdir()):
            if not path.is_file() or not self._is_root_fixture(path):
                continue
            shutil.copy2(path, self.output_dir / path.name)

    def _matching_cbench_project_dir(self) -> Path | None:
        project_root = self.project_dir.parent.parent
        cbench_root = project_root / "CBench"
        if not cbench_root.is_dir():
            return None

        target = _normalized_name(self.project_dir.name)
        for child in sorted(cbench_root.iterdir()):
            if child.is_dir() and _normalized_name(child.name) == target:
                return child
        return None

    def _is_root_fixture(self, path: Path) -> bool:
        if path.name.startswith("."):
            return False
        lowered_name = path.name.lower()
        if lowered_name in ROOT_FIXTURE_SKIP_NAMES:
            return False
        if path.suffix.lower() in ROOT_FIXTURE_SKIP_SUFFIXES:
            return False
        return True

    def _neutralize_unimplemented_bin_mains(self) -> None:
        bin_dir = self.output_dir / "src" / "bin"
        if not bin_dir.is_dir():
            return
        for path in sorted(bin_dir.glob("*.rs")):
            text = path.read_text(encoding="utf-8")
            updated = BIN_MAIN_STUB_RE.sub(r"\g<prefix>\g<suffix>", text)
            if updated != text:
                path.write_text(updated, encoding="utf-8")

    def _index_modules(self) -> None:
        src = self.output_dir / "src"
        if not src.exists():
            return
        for path in sorted(src.rglob("*.rs")):
            if path.name == "lib.rs" or path.parts[-2:-1] == ("bin",) or "interfaces" in path.parts:
                continue
            module = path.stem if path.name != "mod.rs" else path.parent.name
            self.module_files[module] = str(path.relative_to(src))

        interfaces = src / "interfaces"
        if not interfaces.exists():
            return
        for path in sorted(interfaces.glob("*.rs")):
            module = path.stem
            if module in self.module_files:
                continue
            dst = src / f"{module}.rs"
            shutil.copy2(path, dst)
            self.module_files[module] = str(dst.relative_to(src))


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_\s.]+", "", name).lower()
