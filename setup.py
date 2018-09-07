"""
Copyright 2018 University of Freiburg
Joao Carvalho <carvalhj@cs.uni-freiburg.de>
"""
from setuptools import setup
from setuptools.command.install import install


class post_install_command(install):
    def run(self):
        install.run(self)
        """ Download additional nltk packages """
        import nltk
        nltk.download('treebank')
        nltk.download('maxent_treebank_pos_tagger')
        nltk.download('punkt')
        nltk.download('averaged_perceptron_tagger')


setup(
    name="byte_LSTM",
    version="0.0.1",
    author="Joao Carvalho",
    author_email="carvalhj@cs.uni-freiburg.de",
    description="Train and sample LSTM languages models",
    license="MIT",
    packages=['byte_LSTM'],
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    cmdclass={'install': post_install_command}
)
