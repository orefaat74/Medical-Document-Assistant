"""Execute a project notebook with this interpreter and save its real outputs."""

import argparse
import asyncio
import sys
from pathlib import Path

import nbformat
from jupyter_client import KernelManager
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a project notebook and save its real outputs")
    parser.add_argument("notebook", nargs="?", type=Path,
                        default=Path("notebooks/01_data_collection_and_eda.ipynb"))
    args = parser.parse_args()
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    root = Path(__file__).resolve().parents[1]
    path = args.notebook if args.notebook.is_absolute() else root / args.notebook
    notebook = nbformat.read(path, as_version=4)
    manager = KernelManager(kernel_name="python3")
    manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(
        notebook, timeout=180, kernel_name="python3", km=manager,
        resources={"metadata": {"path": str(root)}}, record_timing=False,
    )
    try:
        client.execute()
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
            manager.cleanup_resources()
    nbformat.validate(notebook)
    nbformat.write(notebook, path)
    count = sum(cell.cell_type == "code" for cell in notebook.cells)
    print(f"Executed and validated {count} code cells: notebooks/{path.name}")


if __name__ == "__main__":
    main()
