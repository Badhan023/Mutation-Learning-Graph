We present the Mutation Learning Graph (MLG), a directed graph framework that organizes SARS-CoV-2 variants based on their cumulative mutation profiles relative to the reference genome (NC\_045512.2), thereby capturing the dynamics of mutation propagation.

# Installation
After cloning the git repository, create a conda environment using the environment.yml.
```
cd Mutation-Learning-Graph
conda env create -f environment.yml
```
This will create a conda environment named **mlg**. Activate the **mlg** conda environment and install the dependencies by running the following command.
```
bash install.sh
```

# Building your own dataset
For downloading a dataset, we have used [GISAID](https://gisaid.org/). Go to the homepage, and create an account if you do not have one. Follow the steps below to download a dataset.
1. Log in.
2. EpiCov > Search > Choose a location (preferably a country or a state). Let's assume it to be Bangladesh.
3. For filtering, choose **complete**, **high coverage**, and **collection date complete**.
4. **Select all** the genomes filtered, and click **Download**.
5. When you click Download, you will see five options. Download **Nucleotide Sequences (FASTA)** and **Sequencing technology metadata** separately. Name those **sequences.fasta** and **metadata.tsv** respectively.
6. Create a folder inside the **Mutation-Learning-Graph** directory for the dataset (Bangladesh in this case) and have the fasta and tsv files in that directory.
For any dataset, we will need just these two input files: sequences.fasta and metadata.tsv.

# Run the code
Let us consider the Egypt directory as the test data. The directory has two files: sequences.fasta and metadata.tsv. After installation, run the following command. 
```
bash build_mlg.sh reference.fasta Egypt
```
This will generate several files as the outputs of the whole process. 
To run the baseline models, run the following commands.
```
bash baseline_models.sh Egypt
```
For any other dataset you want to build, replace "Egypt" with your directory name in the above commands.
```
bash build_mlg.sh reference.fasta <dataset_directory>
bash baseline_models.sh <dataset_directory>
```

# MLG dataset
All ten MLG datasets for the regions: Egypt, Iran, Nigeria, Bangladesh, Queensland (Australia), China, Estonia, Wyoming (USA), Chile, and South Africa, can be found [here](https://zenodo.org/records/16952912).
