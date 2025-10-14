# Sys_ML Recommendation System

![GitHub Downloads](https://img.shields.io/github/downloads/zhaoxiaoqi0714/Sys_ML/total?style=for-the-badge)

SysML is a web-based platform designed to simplify robust machine learning workflows for biomedical data. It addresses common challenges like small samples, missing values, and class imbalance by recommending data-adaptive preprocessing and algorithm combinations—backed by systematic benchmarking of hundreds of method pairings. Validated on real-world datasets, SysML boosts both model performance and analytical efficiency, making it a go-to tool for reliable, reproducible biomedicine-focused ML.

## Installation

Step 1. Clone the repo [Git Bash]

```bash
git clone git@github.com:zhaoxiaoqi0714/Sys_ML.git
```
Step 2. Create & activate a conda env named Sys_ML
```bash
conda create -n Sys_ML python=3.8 -y
conda activate Sys_ML
```
Step 3. Install dependencies
```bash
pip install -r requirements.txt
```
## Usage
Step 1. Turn on the workstation
```terminal
python manage.py runserver
```
Step 2. Run the system according to the page instructions

📕Eaxample Input-data/train_matrix.csv/test_matrix.csv
```markdown
| Sample ID                            | ENSG00000000003.15 | ENSG00000000419.13 | ENSG00000000460.17 |
|--------------------------------------|--------------------|--------------------|--------------------|
| TCGA-R6-A8W5-01B-11R-A37I-31        | 5453               | 3707               | 2301               |
| TCGA-2H-A9GK-01A-11R-A37I-31        | 1118               | 1984               | 318                |
| TCGA-L5-A4OF-11A-12R-A260-31        | 1238               | 1065               | 317                |
| TCGA-L5-A4OF-01A-11R-A260-31        | 5229               | 4640               | 1402               |
| TCGA-L5-A88W-01A-11R-A354-31        | 1883               | 3241               | 593                |
| TCGA-IG-A6QS-01A-12R-A336-31        | 1435               | 4068               | 536                |
| TCGA-L5-A8NN-01A-11R-A37I-31        | 2643               | 4636               | 1428               |
| TCGA-JY-A6FB-01A-11R-A336-31        | 940                | 2114               | 921                |
| TCGA-IG-A51D-01A-11R-A36D-31        | 1718               | 5114               | 2322               |
| TCGA-VR-A8ER-01A-11R-A36D-31        | 2203               | 2625               | 572                |
| TCGA-L5-A8NF-01A-11R-A37I-31        | 1464               | 4191               | 640                |
```
📕Eaxample Input-data/train_group.csv/test_group.csv
```markdown
| Sample ID                            | Group   |
|--------------------------------------|---------|
| TCGA-R6-A8W5-01B-11R-A37I-31        | Cancer  |
| TCGA-2H-A9GK-01A-11R-A37I-31        | Cancer  |
| TCGA-L5-A4OF-11A-12R-A260-31        | Normal  |
| TCGA-L5-A4OF-01A-11R-A260-31        | Cancer  |
| TCGA-L5-A88W-01A-11R-A354-31        | Cancer  |
| TCGA-IG-A6QS-01A-12R-A336-31        | Cancer  |
| TCGA-L5-A8NN-01A-11R-A37I-31        | Cancer  |
| TCGA-JY-A6FB-01A-11R-A336-31        | Cancer  |
```
Step 3. Download results
