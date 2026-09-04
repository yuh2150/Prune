import os
import ast
import unittest
from pathlib import Path


class TestArchitectureDependencyRules(unittest.TestCase):
    """
    Architecture Test: Enforces that prune_framework does not import
    legacy root packages (utils, models, apps) except inside designated model loading adapters.
    """

    def test_framework_dependency_isolation(self):
        root_dir = Path(__file__).resolve().parents[2]
        framework_dir = root_dir / "prune_framework"

        forbidden_prefixes = ("utils.", "models.", "apps.", "traffic_analysis", "applications")
        allowed_exceptions = {
            "loader.py",       # ModelLoader delegates to models/transformers
            "yolov5.py",       # YOLOv5Adapter loads weights using models.experimental
        }

        violations = []
        for py_file in framework_dir.glob("**/*.py"):
            if py_file.name in allowed_exceptions:
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=str(py_file))
            except Exception:
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    if any(node.module.startswith(p.rstrip(".")) for p in forbidden_prefixes):
                        violations.append((py_file.name, node.module, node.lineno))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        if any(alias.name.startswith(p.rstrip(".")) for p in forbidden_prefixes):
                            violations.append((py_file.name, alias.name, node.lineno))

        self.assertEqual(
            violations,
            [],
            f"Architecture Boundary Violation! The following framework files illegally import legacy root modules: {violations}"
        )


if __name__ == "__main__":
    unittest.main()
