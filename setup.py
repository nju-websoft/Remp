import io
import os
import re
from sys import version_info
from sysconfig import get_paths
from distutils.core import Extension

from setuptools import find_packages, setup
from pybind11.setup_helpers import Pybind11Extension
from Cython.Build import cythonize


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type("")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="remp",
    version="0.0.1",
    url="https://github.com/nju-websoft/remp",
    license='MIT',

    author="Jiacheng Huang",
    author_email="jchuang.nju@gmail.com",

    description="description",
    long_description=read("README.md"),

    packages=find_packages(exclude=('tests',)),

    ext_modules=cythonize([
        Pybind11Extension(
            "remp.string_matching",
            sources=['remp/ext/string_matching.cpp'],
            extra_compile_args=["-Wno-sign-compare", "-fopenmp", "-O2"],
            extra_link_args=["-fopenmp"]
        ),
        Extension('remp.ssj.jaccard_join_cy', ['remp/ssj/jaccard_join_cy.pyx'], language='c++'),
        Extension('remp.ssj.set_sim_join_cy', ['remp/ssj/set_sim_join_cy.pyx'], language='c++'),
    ]),

    install_requires=[
        'rdflib', 'lxml', 'networkx', 'unidecode', 'numpy', 'pandas', 'scipy', 'py_stringmatching', 'py_stringsimjoin'
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9'
    ],
)
