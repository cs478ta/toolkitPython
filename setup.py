from os import path
from setuptools import Extension, setup

def read_md(file_path):
    with open(file_path, "r") as f:
        return f.read()

setup(name='toolkit',
      version='0.0.1',
      description='CS478 Python Toolkit',
      long_description= "" if not path.isfile("README.md") else read_md('README.md'),
      author="CS478 TA's and Students",
      author_email='cs478ta@cs.byu.edu',
      url='https://github.com/cs478ta/toolkitPython',
      setup_requires=[],
      tests_require=[],
      install_requires=[
          "numpy", "scipy"
      ],
      license=['MIT'],
      packages=['toolkit'],
      scripts=[],
      classifiers=[
          'Development Status :: 2 - Pre-Alpha',
          'Intended Audience :: Science/Research',
          'Natural Language :: English',
          'Operating System :: Windows',
          'Programming Language :: Python',
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.6',
      ],
     )