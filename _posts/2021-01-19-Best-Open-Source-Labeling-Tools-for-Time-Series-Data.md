---
layout: post
title: "Best Open-Source Labeling Tools for Time Series Data"
tagline: 
author: "Wangbo Zheng"
---

Recently, the use and promotion of artificial intelligence have been increasing day by day. To develop AI solutions, more and more data is recorded and stored. Time series data is one of the most important data types besides image data. The IoT sensor record of the factory, the medical single from ICU monitor, ECG signal tracked with your smartwatch are all time-series data. To make better use of these data with machine learning technology, we need to label the data for training of the supervised learning model. 

Data labeling is important but sometimes also very repetitive, cumbersome, and time-consuming. Therefore, several data labeling tools are developed to help people create training datasets for machine learning. Unlike image data, there are not so many labeling tools for time series data. Here are a summary and review of some best available open-source labeling tools for time series data.



## TRAINSET

[TRAINSET](https://trainset.geocene.com/) is a web-based graphical tool for labeling time series data. It evolved from a tool called [SUMSarize](https://github.com/geocene/sumsarizer), which helps facilitate the application of ensemble machine learning tools to time series data. TRAINSET is not only available as a web app, but also can be [installed](https://github.com/Geocene/trainset) and hosted locally. With a brief interface, you can easily select one point or a region for the annotation. It allows you to **add new labels** whenever you want. The visualization can be **zoomed** and **panned**. There is also an **overview** plot of the whole time axis, which can help you move fast or change the focus region. The input format should be CSV. The exported output will be like your input data but with a label column.

<p align="center">
<img src="https://raw.githubusercontent.com/Geocene/trainset/master/TRAINSET-GIF.gif">
</p>

TRAINSET can also handle multi-channel data. However, there are some limitations. The input data should follow the special scheme. Its scheme is like after a pivot transformation. Through, TRAINSET can provide a reference channel, it can still label only **one channel at once**. Also, you can not give one region or point multi labels.

## MNE

[MNE](https://mne.tools/dev/index.html) is an open-source Python package for exploring, visualizing, and analyzing neurophysiological data: MEG, EEG, and more. MNE also provides a method to annotate this **medical signal data**. Because of the character of these signals, MNE can easily assign labels for **multi-channel**. It also supports **multi-label**. Labels can be added programmatically, and you can even pass lists or arrays to the [annotations](https://mne.tools/dev/generated/mne.Annotations.html#mne.Annotations) constructor to label multiple spans at once. In MNE, labels or annotations are list-like objects, where each element comprises three pieces of information: an onset time, a duration, and a description. Annotations can also be created interactively by clicking-and-dragging the mouse in the plot window. You can also **add new labels** in the plot window. 

<p align="center">
<img src="https://mne.tools/dev/_images/sphx_glr_plot_30_annotate_raw_003.png">
</p>

Unlike other labeling tools, MNE can only process medical data like MEG. The input should be also adjusted with the right format. It also requires some Python knowledge to get started. 

## Label Studio

[Label Studio](https://labelstud.io/) is an open-source data labeling and annotation tools for different data types. With Label Studio, you can label image, text, [time-series data](https://labelstud.io/blog/release-080-time-series-labeling.html) using an uncomplicated interface with a standardized output format. Label Studio can be installed with pip or docker. Label Studio supports **CSV** also **JSON** format as input. You can even use an **URL** for online data as input. The output is JSON format with values which contain start and end time for each label. Label Studio provides a brief UI with **overview**, **zoom**, and **pan** function. Label Studio allows you to create labels for **multi-channel synchronized or desynchronized**. It supports also **overlapped annotations**(same or multi-label). 

<p align="center">
<img src="https://labelstud.io/images/release-080/zoom.gif">
</p>



For each labeling project with Label Studio, it requires a **configuration file**. There are many templates available, but some basic **XML** knowledge is still needed to adjust the config file with your data. The name and number of labels are also **fixed** in the config file, which means new labels can not be added during user labeling the data in UI. 


## Conclusion

The following table is a summary of the above three tools. They provide various features and functions. All three tools are recommended based on my experience. Of course, there are some other tools such as [Grafana Labs](https://grafana.com/docs/grafana/latest/dashboards/annotations/), etc. You can decide on one of them or even combine them according to your different project requirements. Hope this article can help you find the right labeling tool. 

|                     | TRAINSET                    | MNE                          | Label Studio     |
| ------------------- | --------------------------- | ---------------------------- | ---------------- |
| Data type           | All time series             | Only neurophysiological data | All time series  |
| Input format        | CSV                         | MNE format                   | CSV, JSON, URL   |
| Install             | npm/ or web without install | pip                          | pip, git, docker |
| Programming require | No                          | Python                       | XML              |
| Zoom, overview      | Yes                         | Yes                          | Yes              |
| Add new label       | Yes                         | Yes, but not in UI           | Yes              |
| Multi-channel       | Yes                         | Yes                          | Yes              |
| Multi-label         | No                          | Yes                          | Yes              |

### References

1. [Geocene/trainset: A very lightweight web application for brushing labels onto time series data; useful for building training sets. (github.com)](https://github.com/Geocene/trainset)

2. [Annotating continuous data — MNE 0.23.dev0 documentation](https://mne.tools/dev/auto_tutorials/raw/plot_30_annotate_raw.html#sphx-glr-auto-tutorials-raw-plot-30-annotate-raw-py)

3. [Label Studio — Data Labeling](https://labelstud.io/blog/release-080-time-series-labeling.html)

4. [machine learning - Interactive labeling/annotating of time series data - Data Science Stack Exchange](https://datascience.stackexchange.com/questions/38080/interactive-labeling-annotating-of-time-series-data)

   


