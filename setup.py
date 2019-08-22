import io
import os
import re

import os
from sys import version_info
from sysconfig import get_paths
from setuptools import find_packages, setup
from distutils.core import Extension
from Cython.Build import cythonize


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())

if os.name == 'nt':
    include_dirs = [os.path.join(get_paths()['platstdlib'], '..', 'Library', 'include')]
    library_dir = os.path.join(get_paths()['platstdlib'], '..', 'Library', 'lib')
    libraries = [os.path.join(library_dir, f.replace('.lib', '')) for f in os.listdir(library_dir) if (f.startswith('boost')) and ('numpy' in f or 'python' in f)]
    extra_compile_args = ['/openmp', '/DBOOST_ALL_NO_LIB']
else:
    include_dirs = [get_paths()['include'] + '/../']
    extra_compile_args = ['-fopenmp', '-std=c++11', '-Wno-sign-compare']
    libraries = ['boost_python%d%d' % (version_info.major, version_info.minor), 'boost_numpy%d%d' % (version_info.major, version_info.minor)]
    

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
        Extension(
            'remp.string_matching',
            sources=['remp/ext/string_matching.cpp'],
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=(['-fopenmp'] if os.name == 'posix' else ['/openmp']),
            libraries=libraries
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
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
