#!/usr/bin/env python
import os
import sys
from os.path import join

import setuptools
from Cython.Distutils import build_ext
from setuptools import Extension, find_packages, setup


here = os.path.abspath(os.path.dirname(__file__))

# To update the package version number, edit pyspectra/__version__.py
version = {}
with open(os.path.join(here, 'pyspectra', '__version__.py')) as f:
    exec(f.read(), version)

with open('README.md') as readme_file:
    readme = readme_file.read()


def search_eigen(hint: str = '/usr/include/eigen3'):
    """Search for the eigen3 library.

    see: http://eigen.tuxfamily.org/index.php?title=Main_Page#Documentation
    """
    if os.path.exists(hint):
        return hint
    else:
        return ""


def search_conda():
    """Search for a conda virtual environment."""
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix is not None:
        conda_include = join(conda_prefix, 'include')
        conda_lib = join(conda_prefix, 'lib')
    else:
        conda_include = ""
        conda_lib = ""
    return conda_include, conda_lib


class get_pybind_include:
    """Helper class to determine the pybind11 include path.

    The purpose of this class is to postpone importing pybind11
    until it is actually installed, so that the ``get_include()``
    method can be invoked.
    """

    def __init__(self, user=False):
        self.user = user

    def __str__(self):
        import pybind11
        return pybind11.get_include(self.user)


def has_flag(compiler, flagname):
    """Return a boolean indicating whether a flag name is supported on the specified compiler.

    As of Python 3.6, CCompiler has a `has_flag` method.
    http: // bugs.python.org/issue26689
    """
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.cc') as f:
        f.write('int main (int argc, char **argv) { return 0; }')
        try:
            compiler.compile([f.name], extra_postargs=[flagname])
        except setuptools.distutils.errors.CompileError:
            return False
    return True


def cpp_flag(compiler):
    """Return the -std=c++[17/14/11] compiler flag.

    The newer version is prefered over c++11 (when it is available).
    """
    flags = ['-std=c++17', '-std=c++14', '-std=c++11']

    for flag in flags:
        if has_flag(compiler, flag):
            return flag

    raise RuntimeError('Unsupported compiler -- at least C++11 support '
                       'is needed!')


class BuildExt(build_ext):
    """A custom build extension for adding compiler-specific options."""

    c_opts = {
        'msvc': ['/EHsc'],
        'unix': [],
    }
    l_opts = {
        'msvc': [],
        'unix': [],
    }

    if sys.platform == 'darwin':
        darwin_opts = ['-stdlib=libc++',
                       '-mmacosx-version-min=10.14', '-fno-sized-deallocation']
        c_opts['unix'] += darwin_opts
        l_opts['unix'] += darwin_opts

    def build_extensions(self):
        """Actual compilation."""
        ct = self.compiler.compiler_type
        opts = self.c_opts.get(ct, [])
        link_opts = self.l_opts.get(ct, [])
        if ct == 'unix':
            opts.append('-DVERSION_INFO="%s"' %
                        self.distribution.get_version())
            opts.append(cpp_flag(self.compiler))
            if has_flag(self.compiler, '-fvisibility=hidden'):
                opts.append('-fvisibility=hidden')
        elif ct == 'msvc':
            opts.append('/DVERSION_INFO=\\"%s\\"' %
                        self.distribution.get_version())
        for ext in self.extensions:
            ext.extra_compile_args = opts
            ext.extra_link_args = link_opts
        build_ext.build_extensions(self)


conda_include, conda_lib = search_conda()
eigen_path = search_eigen()

include_dirs = (
    'include',
    conda_include,
    eigen_path,
    get_pybind_include(),
    get_pybind_include(user=True)
)

library_dirs = [conda_lib]

ext_pybind = Extension(
    'spectra_dense_interface',
    sources=['pyspectra/interface/spectra_dense_interface.cc'],
    include_dirs=list(filter(lambda x: x, include_dirs)),
    library_dirs=list(filter(lambda x: x, library_dirs)),
    language='c++')


setup(
    name='pyspectra',
    version=version['__version__'],
    description="Python interface to the C++ Spectra library",
    long_description=readme + '\n\n',
    author="Netherlands eScience Center",
    author_email='f.zapata@esciencecenter.nl',
    url='https://github.com//pyspectra',
    packages=[
        'pyspectra',
    ],
    include_package_data=True,
    license="Apache Software License 2.0",
    zip_safe=False,
    keywords='pyspectra',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    install_requires=['numpy', "pybind11", "scipy"],
    cmdclass={'build_ext': BuildExt},
    ext_modules=[ext_pybind],
    extras_require={
        'doc': ['sphinx>=2.1',
                'sphinx-autodoc-typehints',
                'sphinx_rtd_theme'
                ],
        'test': ['pytest>=5.4',
                 'pytest-cov'
                 ],
    }
)
