import importlib
import pkgutil
import os

# Automatically import all modules in subdirectories (criteria, pruners, adapters)
_plugins_dir = os.path.dirname(__file__)
for sub_dir in ["criteria", "pruners", "adapters"]:
    sub_path = os.path.join(_plugins_dir, sub_dir)
    if os.path.isdir(sub_path):
        for file in os.listdir(sub_path):
            if file.endswith(".py") and not file.startswith("__"):
                module_name = f"prune_framework.plugins.{sub_dir}.{file[:-3]}"
                importlib.import_module(module_name)
