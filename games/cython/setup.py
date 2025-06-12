from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import numpy
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
print(project_root)

extensions = cythonize(
    [
        Extension(
            # "games.cython.connect4_cython",
            "connect4_cython",
            ["connect4_cython.pyx"],
            language="c++",
            include_dirs=[numpy.get_include(), project_root, "."],
            # include_dirs=[numpy.get_include(), project_root],
            extra_compile_args=[
                "-std=c++17",
                "-Wall",
                "-Wextra",
                "-fdiagnostics-color=always",
            ],
        ),
        Extension(
            # "games.cython.game_cython",
            "game_cython",
            ["game_cython.pyx"],
            language="c++",
            include_dirs=[numpy.get_include(), project_root, "."],
            # include_dirs=[numpy.get_include(), project_root],
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
    name="connect4",
    version="0.1",
    ext_modules=extensions,
)
