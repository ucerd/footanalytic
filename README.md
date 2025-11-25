# FootAnalytic – Foot Weight Analysis System

This repository contains the code and resources for the FootAnalytic project, a system for analysing foot weight distribution using a 32×32 FSR sensor matrix and embedded/host software.

## Structure

- `Cloud/`  
  Tools and scripts for running foot weight analysis in the cloud, including:
  - Example CSV datasets (32×32 pressure maps for left and right feet)
  - Streamlit or similar web-based applications for online visualisation and analysis

- `Real-time/`  
  Real-time acquisition and visualisation tools:
  - `DAQ/` – Data acquisition firmware (e.g. Arduino or embedded code) to read the FSR matrix
  - `Host_GUI/` – Desktop GUI, images, and scripts for real-time heatmaps and reports

- `Papers/`  
  Manuscripts, LaTeX files, and documentation related to the FootAnalytic research work.

## Website

Project website: https://footanalytic.com

## Live Cloud Application for Analysis

http://cloud.pakistansupercomputing.com:8505/
