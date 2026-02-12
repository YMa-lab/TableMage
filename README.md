# TableMage &nbsp; 🧙‍♂️📊

![Python Version](https://img.shields.io/badge/python-3.12-blue)
[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
![Tests Passing](https://github.com/YMa-lab/TableMage/actions/workflows/test.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/tablemage/badge/?version=latest)](https://tablemage.readthedocs.io/en/latest/?badge=latest)

TableMage is a Python package for low-code/conversational clinical data science.
TableMage can help you quickly explore tabular datasets, 
easily perform regression analyses,
and effortlessly benchmark machine learning models.

## Installation

We recommend installing TableMage in a new virtual environment. TableMage supports Python version 3.12.

To install TableMage:
```
git clone https://github.com/YMa-lab/TableMage.git
cd TableMage
pip install .
cd ..
```

## Usage

Please read the demo available on [readthedocs](https://tablemage.readthedocs.io/en/latest/getting_started.html#demo).

> [!NOTE]
> **For MacOS users:** You might run into an error involving [XGBoost](https://xgboost.readthedocs.io/en/stable/#), one of TableMage's dependencies, when using TableMage for the first time.
> To resolve this error, you'll need to install libomp: `brew install libomp`. This requries [Homebrew](https://brew.sh/).

## Updates

- February 2026: Our paper on ChatDA has been published in *npj Artificial Intelligence*! We are working on TableMage's v0.1.0 release. Help us out by reporting bugs in Issues.
- December 2025: We have released a preprint on TableMage's ChatDA agent on *medRxiv*!
- February 2025: We have released an alpha version of TableMage on PyPI!

## Citation

If this software was beneficial in your work, please consider citing it as follows:

```bibtex
@article{Yang2026,
  author = {Yang, Andrew and Woo, Joshua and Zhang, Ryan and Mach, Alan and Ramkumar, Prem and Ma, Ying},
  title = {Tool-wielding language model-based agent offers conversational exploration of clinical tabular data},
  journal = {npj Artificial Intelligence},
  year = {2026},
  volume = {2},
  number = {1},
  pages = {22},
  month = {feb},
  doi = {10.1038/s44387-025-00070-2},
  url = {https://doi.org/10.1038/s44387-025-00070-2},
  issn = {3005-1460},
  abstract = {Advancing evidence-based medicine requires integrating clinical expertise with data analysis. While clinicians contribute essential domain knowledge, applying modern data science methods often requires specialized training, creating a barrier to adoption. To bridge this gap, we developed ChatDA, an artificial intelligence agent enabling large language model-mediated conversational analysis of de-identified clinical tabular datasets. ChatDA empowers clinicians to extract meaningful insights efficiently and accurately, making data-driven clinical research more accessible and effective.}
}
```


