"""Verify copied scientific definitions without importing model dependencies."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
LEGACY_ROOT = Path(os.environ.get("QUASF_LEGACY_ROOT", str(ROOT)))


class ImportLocations(ast.NodeTransformer):
    """Normalize the one allowed import relocation inside copied methods."""

    def visit_ImportFrom(self, node):
        if node.module == "decoding" and node.level == 1:
            node.module, node.level = "evaluate_domain", 0
        return node


def digest(node):
    normalized = ImportLocations().visit(copy.deepcopy(node))
    for part in ast.walk(normalized):
        if hasattr(part, "type_params") and not part.type_params:
            del part.type_params  # Python 3.12 adds empty fields absent in Python 3.10.
    value = ast.dump(normalized, include_attributes=False)
    return hashlib.sha256(value.encode()).hexdigest()


class ReleaseSourceTests(unittest.TestCase):
    """Keep original definitions and prevent experiment-dependent imports."""

    def test_recorded_definition_hashes(self):
        manifest = json.loads((ROOT / "docs/RELEASE_PROVENANCE.json").read_text())
        for item in manifest["files"]:
            text = (ROOT / item["target"]).read_text(encoding="utf-8")
            definitions = {n.name: n for n in ast.parse(text).body
                           if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
            original_path = LEGACY_ROOT / item["source"]
            original = None
            if original_path.exists():
                self.assertIn(hashlib.sha256(original_path.read_bytes()).hexdigest(),
                              {item["source_sha256"], item.get("remote_source_sha256")}, item["source"])
                original_text = original_path.read_text(encoding="utf-8-sig")
                original = {n.name: n for n in ast.parse(original_text).body
                            if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))}
            for name, expected in item["definitions"].items():
                with self.subTest(file=item["target"], definition=name):
                    self.assertIn(name, definitions)
                    self.assertEqual(digest(definitions[name]), expected)
                    if original is not None:
                        self.assertEqual(ast.get_source_segment(original_text, original[name]),
                                         ast.get_source_segment(text, definitions[name]).replace(
                                             "from .decoding import ", "from evaluate_domain import "))

    def test_package_has_no_dynamic_or_experiment_imports(self):
        names = {"model", "train", "common", "decoders", "metrics", "losses",
                 "data", "region_mlp", "lightning_module"}
        for path in (ROOT / "quasf").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, ast.Import):
                    self.assertFalse(any(n.name.startswith("importlib") or n.name == "sys"
                                         for n in node.names), str(path))
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    self.assertNotIn(node.module, names, str(path))
            self.assertNotIn("sys.path", text, str(path))


if __name__ == "__main__":
    unittest.main()
