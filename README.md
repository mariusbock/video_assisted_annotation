# Evaluation of Video-Assisted Annotation of Human IMU Data Across Expertise, Datasets, and Tools

## Abstract
Despite the simplicity of labels and extensive study protocols provided during data collection, the majority of researchers in sensor-based technologies tend to rely on annotations provided by a combination of field experts and researchers themselves. This paper presents a comprehensive study on the quality of annotations provided by expert versus novice annotators for inertial-based activity benchmark datasets. We consider multiple parameters such as the nature of the activities to be labeled, and the annotation tool, to quantify the annotation quality and time needed. 15 participants were tasked to annotate a total of 40 minutes of data from two publicly available benchmark datasets for inertial activity recognition, being simultaneously displayed both video and accelerometer data during annotation. We compare the resulting labels with the ground truth provided by the original dataset authors. Our participants annotated the data using two representative tools. Metrics like F1-Score and Cohen's Kappa showed experience did not ensure better labels. While experts were more accurate on the complex Wetlab dataset (51\% vs 46\%), Novices had 96\% F1 on the simple WEAR dataset versus 92\% for experts. Comparable Kappa scores (0.96 and 0.94 for WEAR, 0.53 and 0.59 for Wetlab) indicated similar quality for both groups, revealing differences in dataset complexity. Furthermore, experts annotated faster regardless of the tool. Given proven success across research, our findings suggest crowdsourcing wearable dataset annotation to non-experts warrants exploration as a valuable yet underinvestigated approach, up to a complexity level beyond which quality may suffer.

## Installation

Clone repository:

```
git clone git@github.com:mariusbock/video_assisted_annotation.git
cd video_assisted_annotation
```

Create [Anaconda](https://www.anaconda.com/products/distribution) environment:

```
conda create -n video_annotation python==3.10.11
conda activate video_annotation
```

Install requirements:
```
pip install -r requirements.txt
```

## Reproduce results

Run the `main.py` file using the created Anaconda enviromnent
```
python main.py
```

## Study data
The individual videos used during each annotation session can be downloaded from [here](https://uni-siegen.sciebo.de/s/2XyAF6wLq8CgEsR). The corresponding inertial sensor data as well as ground truth labels can be found in the `sensor_data` folder.

## Repo Structure
- **annotations**: Folder containing the exported annotations of each subject's annotation sessions using either the MaD-GUI or ELAN-Player for annotation. The annotations were exported using the provided functionality of each tool. For further details on the format of the exported annotation please see the documentation of the [ELAN-Player](https://www.mpi.nl/corpus/html/elan/ch04s03s02.html) and [MaD-GUI](https://mad-gui.readthedocs.io/). Annotations are divided by dataset (WEAR or Wetlab). Each folder contains the annotations of all participants divided by participant and by sessions, i.e. each participant provides two annotation files (each corresponding to a 10 minute labeled segment of the corresponding dataset). The file names follow the structure "sbj_{sbj identifier}_{used annotation tool}_{start minute}_{end minute}"
- **sensor\_data**: Folder containing the raw inertial sensor data of the WEAR and Wetlab dataset used for each of the four types of sessions. The sensor data is by sessions, i.e. each file corresponds to one of the four 10 minute segment of the WEAR or Wetlab dataset, which are to labeled by participants. The file names follow the structure "{dataset}_{start minute}_{end minute}. The CSV files consist of 6 columns that are [record_id, time, acceleration_x, acceleration_y, acceleration_z, label]
- **data\_prep**: Python script used for converting the extracted annotations to CSV format and combine them with the raw ground truth data of each dataset.
- **main.py**: Main python script to recreate all plots and reproduce evaluation results mentioned in the paper.
- **study\_results.xlsx**: Excel sheet containing all per-participant NASA and evaluation metrics results. The sheet is divided into 3 subsheets: "MAD", "ELAN", "Plots". The "MAD" and "ELAN" subsheet contains information on information collected during annotation sessions when participants used the MaD-GUI/ ELAN-Player for annotation. That is the skill level of the participant, pros and cons of the tool (as defined by the participant), time the participant took to annotate the 10-minute segment of the WEAR and Wetlab dataset and NASA-TLX scores. The "MAD" and "ELAN" subsheets further calculate averages and standard deviation across novices, experts and all participants. The "Plots" subsheet provides an overview of all NASA-TLX results, the plot corresponding to Figure 3 of the main paper and two tables containing all calculated per-participant evaluation metrics (F1-score, Cohens-Kappa and NULL-class accuracy) per dataset.
- **nasa\_feedback\_template.pdf**: Post-experiment questionnaire sheet collecting NASA-TLX results and feedback regarding the annotation tools.

## Cite as
Coming soon
