"""Simple test for built-in tools without pytest."""

from pathlib import Path
import tempfile
import json


def main():
    from openagent.tools.builtin import read, write, edit, glob, grep, notebook_edit

    print("=== Testing Built-in Tools ===\n")

    # Test 1: Write a file (sync tool, no await needed)
    print("1. Testing 'write' tool...")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        result = write(str(test_file.resolve()), "Hello, World!\nThis is a test.")
        print(f"   Result: {result}")

    # Test 2: Read the file (sync tool, no await needed)
    print("\n2. Testing 'read' tool...")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        write(str(test_file.resolve()), "Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
        result = read(str(test_file.resolve()))
        print(f"   Full content:\n{result}")

    # Test 3: Read with line range (sync tool, no await needed)
    print("\n3. Testing 'read' with line range...")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        write(str(test_file.resolve()), "Line 1\nLine 2\nLine 3\nLine 4\nLine 5")
        result = read(str(test_file.resolve()), line_start=2, line_end=4)
        print(f"   Lines 2-4:\n{result}")

    # Test 4: Edit a file (sync tool, no await needed)
    print("\n4. Testing 'edit' tool...")
    with tempfile.TemporaryDirectory() as tmpdir:
        test_file = Path(tmpdir) / "test.txt"
        write(str(test_file.resolve()), "Hello World\nHello Python")
        result = edit(str(test_file.resolve()), "World", "Universe")
        print(f"   Result: {result}")
        content = read(str(test_file.resolve()))
        print(f"   After edit:\n{content}")

    # Test 5: Glob pattern search (sync tool, no await needed)
    print("\n5. Testing 'glob' tool...")
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.py").touch()
        (Path(tmpdir) / "main.py").touch()
        (Path(tmpdir) / "readme.md").touch()
        result = glob("*.py", path=tmpdir)
        print(f"   Python files:\n{result}")

    # Test 6: Grep search (sync tool, no await needed)
    print("\n6. Testing 'grep' tool...")
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / "test.txt").write_text("Hello World\nFoo Bar\nBaz Qux")
        result = grep("World", path=tmpdir)
        print(f"   Search results:\n{result}")

    # Test 7: Notebook edit (sync tool, no await needed)
    print("\n7. Testing 'notebook_edit' tool...")
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.ipynb') as f:
        nb_content = {
            "cells": [
                {"cell_type": "code", "source": ["print('hello')"]},
                {"cell_type": "markdown", "source": ["# Title"]},
            ],
            "metadata": {},
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        json.dump(nb_content, f)
        path = Path(f.name).resolve()

    result = notebook_edit(str(path), cell_index=0, new_source="print('world')")
    print(f"   Result: {result}")

    # Verify the change
    with open(path) as f:
        nb = json.load(f)
    print(f"   Cell 0 source after edit: {nb['cells'][0]['source']}")

    Path(path).unlink()

    print("\n=== All tests completed! ===")


if __name__ == "__main__":
    main()  # Run synchronously since tools are sync functions
