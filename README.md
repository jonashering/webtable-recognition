# webtable-recognition

> Building knowledge from web tables has attracted a significant interest in research in the past 10 years. With ever growing web data, the challenge as well as the potential to utilize these massive, crowd-sourced information for information extraction is more present than ever. However, the manifold types, shapes, and purposes with which web tables are used throughout the web renders processing them a challenge. We propose a novel classification approach, which categorizes web tables based on their visual properties using deep neural networks. This approach can be integrated into information extraction pipelines to facilitate a more appropriate and accurate processing of web tables.


This repository contains and documents our work in the master's seminar ["Processing Web Tables"](https://hpi.de/naumann/teaching/teaching/ss-19/processing-web-tables.html) at Hasso Plattner Institute in the summer term 2019.

We have studied the question of how the different types of web tables can be reliably identified. For this purpose we have developed a prototype that has been implemented in this repository. Additionally, we implemented a baseline solution to evaluate the performance of our approach. This baseline is based on the paper ["Web-scale table census and classification"](https://dl.acm.org/citation.cfm?id=1935826.1935904) by Eric Crestan and Patrick Pantel.


## Project structure

The modules of our pipeline are implemented in `wtrec`. Our pipeline looks like this:

```
(Dataset)
    |
    |
[loader.py] -> [transformer.py] -> [splitter.py] -> [classifier.py] -> [evaluator.py]
```

The actual pipelines are implemented using Jupyter Notebooks: see `approach_pipeline.ipynb` and `baseline_pipeline.ipynb` for details.


## Setup

Make sure to install Python >= 3.6 and the Python package manager pip3.

To install the project requirements, run:
```
pip3 install -r requirements.txt
# Plus, on Ubuntu run:
sudo apt-get install wkhtmltopdf
# Or, on MacOS run:
brew install wkhtmltopdf
```

## Dataset

We crawled and labelled our own web table dataset for the task of web table type classification. Download the dataset from the [Releases](https://github.com/jonashering/webtable-recognition/releases) section, and unzip it in the `./data` directory.

## Run pipelines

As mentioned, our pipelines are implemented using Jupyter Notebooks. To start a pipeline run:
```
jupyter notebook
# open a web browser on the URL shown in your terminal
# open the approach_pipeline or baseline_pipeline notebook in the web interface
# run all cells!
``` 


## Report

Find our report at `report.pdf`.
