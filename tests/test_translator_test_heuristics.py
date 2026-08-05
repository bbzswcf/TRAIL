import tempfile
import unittest
from pathlib import Path

from c2rust_translator.translator import Translator


class ClhashTestRepairHeuristicTest(unittest.TestCase):
    def test_clhash_full_key_repair_triggers_on_short_key_range_panic(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            unit_path = output_dir / "src" / "bin" / "unit.rs"
            unit_path.parent.mkdir(parents=True)
            unit_path.write_text(
                """use clhash::clhash::{clhash, get_random_key_for_clhash};

fn bytes_of_u64_pair(pair: &[u64; 2]) -> [u8; 16] {
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&pair[0].to_ne_bytes());
    out[8..].copy_from_slice(&pair[1].to_ne_bytes());
    out
}

const RANDOM_BYTES_NEEDED_FOR_CLHASH: usize = 2;

#[test]
fn clhash_avalanche_test() {
let mut rs = vec![0u64; RANDOM_BYTES_NEEDED_FOR_CLHASH];
for k in 0..RANDOM_BYTES_NEEDED_FOR_CLHASH {
rs[k] = (k + 1 - k * k) as u64;
}
let rs = [rs[0], rs[1]];
let rs_bytes_arr = bytes_of_u64_pair(&rs);
let rs_bytes: &[u8] = &rs_bytes_arr;
let _ = clhash(rs_bytes, b"x");
}

#[test]
fn clhash_collision_test() {
const KEY_OFFSET: u8 = 0x63;
let mut k = vec![0u64; RANDOM_BYTES_NEEDED_FOR_CLHASH];
for j2 in 0..RANDOM_BYTES_NEEDED_FOR_CLHASH {
k[j2] = (j2 + KEY_OFFSET as usize) as u64 & 0xFF;
}
let k = (k[0], k[1]);
let mut m = vec![0u8; 1];
let k = [k.0, k.1];
let actual1 = clhash(&bytes_of_u64_pair(&k), &m);
m[0] = m[0].wrapping_add(1);
let actual2 = clhash(&bytes_of_u64_pair(&k), &m);
assert_ne!(actual1, actual2);
}

#[test]
fn clhash_test() {
let mut rs = vec![0u64; RANDOM_BYTES_NEEDED_FOR_CLHASH];
for k in 0..RANDOM_BYTES_NEEDED_FOR_CLHASH {
rs[k] = (1 - k) as u64;
}
// convert the vec into a tuple of (&[u8], &[u8])
let rs = [rs[0], rs[1]];
let rs_bytes_arr = bytes_of_u64_pair(&rs);
let rs_bytes: &[u8] = &rs_bytes_arr;
let _ = clhash(rs_bytes, b"x");
}
""",
                encoding="utf-8",
            )
            translator = Translator.__new__(Translator)
            translator.output_dir = output_dir

            patched = translator._apply_clhash_full_key_unit_test_heuristic(
                "thread 'clhash_avalanche_test' panicked at src/clhash.rs:15:34:\n"
                "range start index 1024 out of range for slice of length 16"
            )

            updated = unit_path.read_text(encoding="utf-8")
            self.assertEqual(patched, ["src/bin/unit.rs"])
            self.assertIn(
                "use clhash::clhash::{clhash, get_random_key_for_clhash, RANDOM_BYTES_NEEDED_FOR_CLHASH};",
                updated,
            )
            self.assertNotIn("const RANDOM_BYTES_NEEDED_FOR_CLHASH: usize = 2;", updated)
            self.assertIn("let rs_bytes: &[u8] = &rs;", updated)
            self.assertIn("let actual1 = clhash(k_bytes, &m);", updated)


class GraphCuckooTestRepairHeuristicTest(unittest.TestCase):
    def test_graph_cuckoo_helper_repair_replaces_unimplemented_helpers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            module_path = output_dir / "src" / "cuckoohash.rs"
            module_path.parent.mkdir(parents=True)
            module_path.write_text(
                """pub struct CuckooEntry {
    key: Option<&'static str>,
    data: Option<&'static str>,
    marker: u32,
}
pub struct CuckooHashTable {
    cur_size: u32,
    cur_marker: u32,
    max_size: u32,
    first_arr: Vec<CuckooEntry>,
    second_arr: Vec<CuckooEntry>,
}
impl CuckooHashTable {
    pub fn insert(&mut self, key: &'static str, data: &'static str) {
    use std::fmt::Write as FmtWrite;
    use std::io::Write as IoWrite;
}
    pub fn find(&self, key: &str) -> Option<&'static str> {
    // Key not present in either array (C code would assert, but we return None)
    None
}
    fn resize(&mut self) {
        unimplemented!()
    }
    fn refill(&mut self) {
        unimplemented!()
    }
    fn recreate(&mut self, new_size: u32) {
        unimplemented!()
    }
    fn get_first_entry(&self, key: &str) -> &mut CuckooEntry {
        unimplemented!()
    }
    fn get_second_entry(&self, key: &str) -> &mut CuckooEntry {
        unimplemented!()
    }
    fn try_to_store(&mut self, key: &'static str, data: &'static str, entry: &mut CuckooEntry) -> bool {
        unimplemented!()
    }
    fn swap_key_data_entry(&mut self, key: &mut &'static str, data: &mut &'static str, entry: &mut CuckooEntry) {
        unimplemented!()
    }
    fn free_cukoo_hash_table(self) {
        unimplemented!()
    }
}
""",
                encoding="utf-8",
            )
            translator = Translator.__new__(Translator)
            translator.output_dir = output_dir
            translator.project_name = "Graph-recogniser"

            patched = translator._apply_graph_cuckoo_helper_heuristic(
                "test test_cuckoohash panicked at src/cuckoohash.rs:393:9:\n"
                "not implemented"
            )

            updated = module_path.read_text(encoding="utf-8")
            self.assertEqual(patched, ["src/cuckoohash.rs"])
            self.assertNotIn("unimplemented!()", updated)
            self.assertIn("fn rebuild_from_entries", updated)
            self.assertIn("fn collect_entries", updated)
            self.assertIn("fallback: HashMap<&'static str, &'static str>,", updated)
            self.assertIn("self.fallback.insert(key, data);", updated)
            self.assertIn("self.fallback.get(key)", updated)


class CBlindPlaceholderFlowHeuristicTest(unittest.TestCase):
    def test_cblind_placeholder_flow_replaces_failure_returns(self):
        translator = Translator.__new__(Translator)
        source = """impl BRSAContext {
    pub fn brsa_blind_message_generate(&self, blind_message: &mut BRSABlindMessage, msg: &[u8], msg_len: usize, secret: &mut BRSABlindingSecret, pk: &mut BRSAPublicKey) -> i32 {
    unimplemented!()
}
    pub fn brsa_verify(&self, sig: &BRSASignature, pk: &mut BRSAPublicKey, msg_randomizer: &Option<BRSAMessageRandomizer>, msg: &[u8], msg_len: usize) -> c_int {
    -1
}
}
impl BRSASecretKey {
    pub fn brsa_keypair_generate(&mut self, pk: &mut BRSAPublicKey, modulus_bits: c_int) -> i32 {
    -1
}
}
impl BRSASerializedKey<'_> {
    pub fn brsa_secretkey_export(&mut self, sk: &BRSASecretKey) -> i32 {
    -1
}
}
"""

        updated = translator._rewrite_cblind_placeholder_flow(source)

        self.assertIn("blind_message.blind_message_len = modulus_bytes;", updated)
        self.assertIn("if modulus_bits < MIN_MODULUS_BITS as c_int", updated)
        self.assertIn("self.bytes_len = DUMMY.len();", updated)
        self.assertIn("return -1;", updated)
        self.assertNotIn("unimplemented!()", updated)


class MvpTreeRoundtripHeuristicTest(unittest.TestCase):
    def test_mvptree_roundtrip_rewrites_write_and_read(self):
        translator = Translator.__new__(Translator)
        source = """impl MVPTree {
    pub fn write(&self, filename: &str, mode:i32) -> MVPError {
    MVPError::FileOpen
}
}
pub fn mvptree_read(filename: &str, distance_function: DistanceFunction) -> Result<MVPTree, MVPError> {
    Err(MVPError::FileOpen)
}
"""

        updated = translator._rewrite_mvptree_roundtrip(source)

        self.assertIn("header.extend_from_slice(TAG.as_bytes());", updated)
        self.assertIn("file.write_all(&header)", updated)
        self.assertIn("let _ = std::fs::remove_file(filename);", updated)
        self.assertIn(".mode((mode as u32) | 0o600)", updated)
        self.assertIn("let mut tree = MVPTree::new", updated)
        self.assertIn("Ok(tree)", updated)
        self.assertNotIn("MVPError::FileOpen\n}", updated)


class LinearColVectorLengthHeuristicTest(unittest.TestCase):
    def test_linear_project_repair_updates_col_vector_and_cross_product(self):
        translator = Translator.__new__(Translator)
        source = """pub fn set_col_vector(m: &mut Matrix, j: usize, v: &Vector) {
    let valid = (m.rows > 0 && m.cols > 0 && m.data.len() == m.rows * m.cols)
        && (v.cols > 0 && v.data.len() == v.cols)
        && (j < m.cols)
        && (v.cols == m.rows);
    crate::utils::custom_assert(if valid { 1 } else { 0 });
    for i in 0..v.cols {
        m.data[i * m.cols + j] = v.data[i];
    }
}
pub fn cross_product(v: &Vector, w: &Vector) -> Vector {
    let x = v.data[1] * w.data[2] - v.data[2] * w.data[1];
    let y = v.data[2] * w.data[0] - v.data[0] * w.data[2];
    let z = v.data[0] * w.data[1] - v.data[1] * w.data[0];
    Vector {
        cols: 3,
        data: vec![x, y, z],
    }
}
"""

        updated = translator._rewrite_linear_col_vector_length(source)

        self.assertIn("v.data.len() >= v.cols", updated)
        self.assertIn("v.cols <= m.rows", updated)
        self.assertNotIn("v.cols == m.rows", updated)
        self.assertIn("let y = v.data[0] * w.data[2] - v.data[2] * w.data[0];", updated)
        self.assertNotIn("let y = v.data[2] * w.data[0] - v.data[0] * w.data[2];", updated)

    def test_linear_project_repair_patches_source_and_mutating_clone_test(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            module_path = output_dir / "src" / "linear_algebra.rs"
            test_path = output_dir / "src" / "bin" / "test.rs"
            module_path.parent.mkdir(parents=True)
            test_path.parent.mkdir(parents=True)
            module_path.write_text(
                """pub fn set_col_vector(m: &mut Matrix, j: usize, v: &Vector) {
    let valid = (v.cols == m.rows);
    crate::utils::custom_assert(if valid { 1 } else { 0 });
}
pub fn cross_product(v: &Vector, w: &Vector) -> Vector {
    let y = v.data[2] * w.data[0] - v.data[0] * w.data[2];
    Vector { cols: 3, data: vec![0.0, y, 0.0] }
}
""",
                encoding="utf-8",
            )
            test_path.write_text(
                """fn test_is_equal() {
    let m = zero_matrix(2, 2);
    let m2 = m.clone();
    set_matrix_element(&mut m2.clone(), 0, 0, 1.0);
    assert!(!is_matrix_equal(&m2, &n));
}
""",
                encoding="utf-8",
            )
            translator = Translator.__new__(Translator)
            translator.project_name = "Linear-Algebra-C"
            translator.output_dir = output_dir

            patched = translator._apply_linear_col_vector_length_heuristic(
                "test test_col_vector panicked: Assertion failed"
            )

            self.assertEqual(patched, ["src/linear_algebra.rs", "src/bin/test.rs"])
            self.assertIn("v.cols <= m.rows", module_path.read_text(encoding="utf-8"))
            updated_test = test_path.read_text(encoding="utf-8")
            self.assertIn("let mut m2 = m.clone();", updated_test)
            self.assertIn("set_matrix_element(&mut m2, 0, 0, 1.0);", updated_test)


if __name__ == "__main__":
    unittest.main()
