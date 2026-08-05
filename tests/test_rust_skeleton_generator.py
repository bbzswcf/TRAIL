import tempfile
import unittest
from pathlib import Path

from c2rust_translator.rust_skeleton_generator import Skeleton


class RustSkeletonGeneratorTest(unittest.TestCase):
    def test_copies_matching_cbench_root_fixtures_without_source_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            rbench = root / "data" / "project" / "RBench" / "Simple_Config"
            cbench = root / "data" / "project" / "CBench" / "Simple-Config"
            (rbench / "src").mkdir(parents=True)
            cbench.mkdir(parents=True)
            (rbench / "Cargo.toml").write_text("[package]\nname='demo'\n", encoding="utf-8")
            (rbench / "src" / "lib.rs").write_text("", encoding="utf-8")
            (cbench / "sample.cfg").write_text("font: demo\n", encoding="utf-8")
            (cbench / "config.c").write_text("int main(void) { return 0; }\n", encoding="utf-8")
            (cbench / "README.md").write_text("docs\n", encoding="utf-8")

            output = root / "output"
            Skeleton(str(rbench), str(output)).generate()

            self.assertEqual((output / "sample.cfg").read_text(encoding="utf-8"), "font: demo\n")
            self.assertFalse((output / "config.c").exists())
            self.assertFalse((output / "README.md").exists())

    def test_neutralizes_unimplemented_binary_main_without_touching_interface_stubs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            project = root / "project"
            (project / "src" / "bin").mkdir(parents=True)
            (project / "src" / "interfaces").mkdir(parents=True)
            (project / "src" / "lib.rs").write_text("pub mod api;\n", encoding="utf-8")
            (project / "src" / "bin" / "benchmark.rs").write_text(
                "pub fn helper() -> i32 { 1 }\npub fn main() {\n    unimplemented!()\n}\n",
                encoding="utf-8",
            )
            (project / "src" / "interfaces" / "api.rs").write_text(
                "pub fn api() -> i32 {\n    unimplemented!()\n}\n",
                encoding="utf-8",
            )

            output = root / "output"
            Skeleton(str(project), str(output)).generate()

            benchmark = (output / "src" / "bin" / "benchmark.rs").read_text(encoding="utf-8")
            interface = (output / "src" / "interfaces" / "api.rs").read_text(encoding="utf-8")
            copied_module = (output / "src" / "api.rs").read_text(encoding="utf-8")

        self.assertNotIn("unimplemented!()", benchmark)
        self.assertIn("pub fn main() {", benchmark)
        self.assertIn("unimplemented!()", interface)
        self.assertIn("unimplemented!()", copied_module)


if __name__ == "__main__":
    unittest.main()
