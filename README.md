A simple SSVEP classification pipeline for the Wang benchmark EEG dataset using Canonical Correlation Analysis (CCA).

## What this code does

This script loads EEG data from the Wang SSVEP dataset, extracts a time window from each trial, optionally preprocesses the signal, generates sinusoidal reference signals for each target frequency, and uses CCA to predict which visual stimulus frequency the subject was attending to.

It then evaluates classification accuracy across all targets, blocks, and subjects in the dataset.

## Dataset

This code expects the Wang dataset files:

- `S1.mat` to `S35.mat`
- `Freq_Phase.mat`
- `64-channels.loc`

Place all of these inside one dataset folder.

Dataset link:  
[Google Drive dataset file](https://drive.google.com/file/d/1xGu1LDg_dOutafB0VI7VzHKW7Y0ohD34/view?usp=drive_link)

Set the environment variable `WANG_DATASET_DIR` to that folder before running the script.

## Requirements

Install dependencies with:

```bash
pip install numpy scipy scikit-learn