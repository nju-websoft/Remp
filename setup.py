import io
import os
import re

from sys import version_info
from setuptools import find_packages
from setuptools import setup
from distutils.core import Extension


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
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

    ext_modules=[
        Extension(
            'remp.string_matching',
            sources=['remp/ext/string_matching.cpp', 'remp/ext/array_bigram_jaccard.cpp'],
            extra_compile_args=['-fopenmp', '-Wno-sign-compare'],
            extra_link_args=['-fopenmp'],
            libraries=['boost_python', 'boost_numpy%d%d' % (version_info.major, version_info.micro)]
        )
    ],

    install_requires=[
        'rdflib', 'lxml', 'networkx', 'unidecode', 'numpy', 'pandas', 'scipy', 'py_stringmatching', 'py_stringsimjoin'
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
)
