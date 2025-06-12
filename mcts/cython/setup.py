# setup.py
from setuptools import setup
from Cython.Build import cythonize
from setuptools import Extension
import numpy
import os

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
games_cython_path = os.path.join(project_root, "games/cython")
print(project_root)

extensions = cythonize(
    [
        Extension(
            "mcts.cython.node",
            ["node.pyx"],
            language="c++",
            include_dirs=[
                numpy.get_include(),
                project_root,
                games_cython_path,
                # "../../../AlphaZero",
                # ".",
            ],
            extra_compile_args=[
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-fdiagnostics-color=always",
            ],
        ),
    ],
    compiler_directives={
        "language_level": "3",
        "binding": True,  # Optional: improves subclassing behavior
    },
    force=True,
    gdb_debug=True,
    emit_linenums=True,
)

setup(
    name="node",
    version="0.1",
    ext_modules=extensions,
)
