# Comparison of EEG Signal Processing and Analysis Methods for Autism Spectrum Disorder Detection

This is a repository containing supplementary code for "Comparison of EEG Signal Processing and Analysis Methods for Autism Spectrum Disorder Detection" bachelor thesis by Anna Kachmarchyk. 

>[!IMPORTANT]
> The dataset provided by "NEUROLOGY" center used in this research contains sensitive medical information and is not redistributable, therefore we did not upload it on purpose. Notebook have been **intentionally pre-populated with anonymized results of execution** to mitigate this issue. However, **it is not possible to run the notebooks without the dataset and the anonymizer code**. To get access, contact us. 

## File structure

- `whole_night_dataset_preprocessing.py`: dataset filtering for the overnight EEG recording dataset provided by "NEUROLOGY" center. It looks for background labels in descriptions, checks channels and filters out recordings accordingly.
- `whole_night_model_benchmark.ipynb`: training and evaluation of EEGNet classifier on the same dataset. It builds upon preceeding research, replicates it and shows inconsistencies by using CV with LOSO and trying different combinations of the test pair.
- `dataset_preprocessing_final.ipynb`: preprocessing and filtering of datasets, and creation of files with filtered data. Used to create inputs for `experiments_final.ipynb`
- `experiments_final.ipynb`: training and evaluation on the variations of the 150 second EEG datasets with filtering methods applied: original, [sheffield dataset](https://orda.shef.ac.uk/articles/dataset/EEG_Data_for_Electrophysiological_signatures_of_brain_aging_in_autism_spectrum_disorder_/16840351) and a combination of both.
