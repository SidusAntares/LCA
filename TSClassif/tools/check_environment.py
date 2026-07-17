#!/usr/bin/env python3
"""Audit the existing offline environment without installing anything."""

import importlib
import platform
import sys


DEPENDENCIES = {
    "torch": (True, "required; cannot be replaced inside this project"),
    "numpy": (True, "required; cannot be replaced inside this project"),
    "pandas": (True, "required for experiment result tables"),
    "sklearn": (True, "required for stateless metrics"),
    "einops": (True, "required by models/LCA_models.py"),
    "torchmetrics": (False, "optional; replaced by stateless sklearn metrics"),
    "torchvision": (False, "optional; replaced by pure-PyTorch channel normalization"),
    "wandb": (False, "optional; local training does not use online logging"),
}


def main():
    print(f"Python: {platform.python_version()} ({sys.executable})")
    imported = {}
    required_failures = []
    optional_missing = []
    for name, (required, mitigation) in DEPENDENCIES.items():
        try:
            module = importlib.import_module(name)
            imported[name] = module
            version = getattr(module, "__version__", "unknown")
            print(f"dependency {name}: OK ({version})")
        except Exception as exc:
            print(f"dependency {name}: MISSING/ERROR ({type(exc).__name__}: {exc})")
            if required:
                required_failures.append((name, mitigation))
            else:
                optional_missing.append((name, mitigation))

    torch = imported.get("torch")
    api_failures = []
    if torch is None:
        print("PyTorch: unavailable")
        print("CUDA runtime: unavailable")
        print("GPU list: unavailable")
        api_failures.extend(["torch.func", "torch.func.vmap", "torch.func.jacfwd"])
    else:
        print(f"PyTorch: {torch.__version__}")
        print(f"CUDA runtime: {torch.version.cuda}")
        cuda_available = torch.cuda.is_available()
        device_count = torch.cuda.device_count()
        print(f"CUDA available: {cuda_available}")
        print(f"GPU count: {device_count}")
        for index in range(device_count):
            props = torch.cuda.get_device_properties(index)
            memory_mb = props.total_memory / (1024 ** 2)
            print(f"GPU {index}: {props.name} ({memory_mb:.0f} MiB)")
        func = getattr(torch, "func", None)
        api_checks = {
            "torch.func": func is not None,
            "torch.func.vmap": func is not None and hasattr(func, "vmap"),
            "torch.func.jacfwd": func is not None and hasattr(func, "jacfwd"),
        }
        for api_name, available in api_checks.items():
            print(f"LCA API {api_name}: {'OK' if available else 'MISSING'}")
            if not available:
                api_failures.append(api_name)
        if not cuda_available or device_count < 1:
            required_failures.append(("CUDA GPU", "required for the requested GPU experiment"))

    if optional_missing:
        print("Optional missing dependencies and project-local avoidance:")
        for name, mitigation in optional_missing:
            print(f"- {name}: {mitigation}")
    if required_failures:
        print("Required dependency failures:")
        for name, mitigation in required_failures:
            print(f"- {name}: {mitigation}")
    if api_failures:
        print("Missing LCA APIs: " + ", ".join(api_failures))

    passed = not required_failures and not api_failures
    print("PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

