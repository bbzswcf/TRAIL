import tempfile
import unittest
from pathlib import Path

from c2rust_translator.rbench_interface import InterfaceIndex


class InterfaceIndexTest(unittest.TestCase):
    def test_lists_public_callables_without_expanding_translation_stub_set(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            interfaces = Path(tmpdir) / "src" / "interfaces"
            interfaces.mkdir(parents=True)
            (interfaces / "buffer.rs").write_text(
                """
pub struct Buffer;

impl Buffer {
    pub fn new(size: usize) -> Self {
        Buffer
    }

    pub fn push(&mut self) {
        unimplemented!()
    }
}
""",
                encoding="utf-8",
            )

            index = InterfaceIndex(tmpdir).build()

        self.assertEqual([item.name for item in index.list_unimplemented()], ["push"])
        self.assertEqual([item.name for item in index.list_public()], ["new", "push"])

    def test_trait_impl_stubs_are_translated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            interfaces = Path(tmpdir) / "src" / "interfaces"
            interfaces.mkdir(parents=True)
            (interfaces / "hasher.rs").write_text(
                """
pub struct Hasher;

impl Drop for Hasher {
    fn drop(&mut self) {
        unimplemented!()
    }
}
""",
                encoding="utf-8",
            )

            index = InterfaceIndex(tmpdir).build()

        self.assertEqual([item.name for item in index.list_unimplemented()], ["drop"])
        self.assertEqual([item.name for item in index.list_public()], ["drop"])


if __name__ == "__main__":
    unittest.main()
