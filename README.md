We present the Mutation Learning Graph (MLG), a directed graph framework that organizes SARS-CoV-2 variants based on their cumulative mutation profiles relative to the reference genome (NC\_045512.2), thereby capturing the dynamics of mutation propagation.
MLG datasets of 10 geographic locations can be found [here](https://zenodo.org/records/16952912).

# Installation
After git cloning, create a conda environment using the environment.yml.
```
conda env create -f environment.yml
```
This will create a conda environment named **mlg**. Activate the **mlg** conda environment and install the dependencies by running the following command.
```
bash install.sh
```

# Dataset
For downloading a dataset, we have used [GISAID](https://gisaid.org/). Go to the homepage, and create an account if you do not have one. Follow the steps below to download a dataset.
1. Log in.
2. EpiCov > Search > Choose a location (preferably a country or a state). Let's assume it to be Bangladesh.
3. For filtering, choose **complete**, **high coverage**, and **collection date complete**.
4. **Select all** the genomes filtered, and click **Download**.
5. When you click Download, you will see five options. Download **Nucleotide Sequences (FASTA)** and **Sequencing technology metadata** separately. Name those **sequences.fasta** and **metadata.tsv** respectively.
6. Create a folder inside the **Mutation-Learning-Graph** directory for the dataset (Bangladesh in this case) and have the fasta and tsv files in that directory.
   
