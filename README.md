JediLuke's Tensorflow playground
=============================

Most of the code in this repo isn't mine, except for gen_clas.py

gen_clas.py
---------------

gen_clas is a general purpose classifier written in  tensorflow. It can classify different datasets of varying length, dimension and number of categories, with no changes to the python code. Using the .csv's in this repo, I have classified the MNIST, iris and blood transfusion datasets, all taken from kaggle.

csv format
---------------

First row of CSV is the header

	num_data_rows, data_dimensionality, (optional) add category names

Remaining rows are the data itself, the last column is the category that row should be classified as

	dimension_1, dimension_2, ..., dimension_n, category(integer)


MNIST
---------------

Github hates large files, so I can't upload MNIST. Transforming traditional MNIST into the format for this program is left as an exercise to the reader ;)


Installation
---------------

See: http://www.heatonresearch.com/2017/01/01/tensorflow-windows-gpu.html

* Install CUDA drivers - http://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/#axzz4bGDH1zNh
* $ pip3 install tensorflow-gpu