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
- **annotations**: Folder containing the exported annotations of each subject's annotation sessions using either the MAD-GUI or ELAN-Player for annotation.
- **sensor\_data**: Folder containing the raw inertial sensor data of the WEAR and Wetlab dataset used for each of the four types of sessions.
- **data\_prep**: Python script used for converting the extracted annotations to CSV format and combine them with the raw ground truth data of each dataset.
- **main.py**: Main python script to recreate all plots and reproduce evaluation results mentioned in the paper.
- **study\_results.xlsx**: Excel sheet containing all per-participant NASA and evaluation metrics results.
- **nasa\_feedback\_template.pdf**: Post-experiment questionnaire sheet collecting NASA-TLX results and feedback regarding the annotation tools.

## Cite as
Coming soon