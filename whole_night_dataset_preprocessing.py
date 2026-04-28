
import os
import os.path as op
import mne
import numpy as np
import pandas as pd

base_dir = './'
data_dir = op.join(base_dir, 'ASD_data_copy')
# windows_dataset_dir = op.join(data_dir, 'windows_dataset')
dataset_dir = op.join(base_dir, 'dataset_expanded') # Path to the dataset folder
windows_dataset_dir = op.join(base_dir, 'windows_dataset_expanded') # Path to the windows_dataset folder


def read_edfs(data_dir):
    data_dir_autism = op.join(data_dir, 'Autism')
    data_dir_norm = op.join(data_dir, 'Norm')

    subject_id = 1

    def read_edfs_from_dir(dir, label):
        records = []
        for file in os.listdir(dir):
            # check only EDF files
            if file.lower().endswith('.edf'):
                nonlocal subject_id

                raw = mne.io.read_raw_edf(op.join(dir, file), stim_channel='auto') # Returns a Raw object containing EEG data
                # raw.load_data() # Loading continuous data
                events_from_annot, event_dict = mne.events_from_annotations(raw) # Get events and event_id from an Annotations object.
                print(event_dict)

                records.append({'file': file,
                                'subject_id': subject_id,
                                'raw': raw,
                                'events_from_annot': events_from_annot,
                                'event_dict': event_dict,
                                'label': label })

                subject_id += 1

        return records
    
    records = []
    records.extend(read_edfs_from_dir(data_dir_autism, 'autism'))
    records.extend(read_edfs_from_dir(data_dir_norm, 'norm'))

    return records


def main():
    background_keys = ["Bgrnd", "Background", "Фоновая запись(testUser)", "Фоновий запис(testUser)(testUser)", "Фоновая запись (testUser)" ]

    records = read_edfs(data_dir)
    print('\n\n')
    # print(records)


    print(f"len(records)={len(records)}")
    print("\n")
    for r in records:
        raw = r['raw']
        print(f"label={r['label']} file={r['file']} \n   ch_names = {raw.info['ch_names']} \n   event_dict={r['event_dict']}")


    # Filter the records by some criterion
    records_filtered = [
    r for r in records
    if any(key in r['event_dict'].keys() for key in background_keys)
]


    if False:
        r_by_label = {'norm': [], 'autism': []}
        for r in records_filtered:
            r_by_label[r['label']].append(r)
        records_filtered = r_by_label['norm'][:1] + r_by_label['autism'][:1]

    for r in records_filtered:
        print(f"{r['raw'].info} {r['label']}")




    # Create dataset and windows_dataset
    import mne
    from braindecode.datasets import BaseDataset, BaseConcatDataset, create_from_mne_raw
    from braindecode.preprocessing.windowers import create_windows_from_events

    sfreq = 250 # Hz
    window_duration = 4 # s
    window_size_samples = window_duration*sfreq

    # parts = [r['raw'] for r in records_filtered]
    parts = []
    for r in records_filtered:
        raw = r['raw'].copy()

        desc = 'Bgrnd_autism' if r['label'] == 'autism' else 'Bgrnd'
        dur = raw.n_times / float(raw.info['sfreq'])

        raw.set_annotations(
            mne.Annotations(
                onset=[0.0],
                duration=[dur],
                description=[desc],
            )
        )

        parts.append(
            raw.drop_channels(['ECG ECG', 'MA 5: MA(build'], on_missing='warn')
        )



    # descriptions = [{"event_code": 0 if r['label'] == 'norm' else 1, "subject": r['subject_id']} for r in records_filtered]
    descriptions = [
    {
        "event_code": 0 if r["label"] == "norm" else 1,
        "subject": r["subject_id"],
        "file": r["file"],                                     # <- add this
        "filepath": op.join(data_dir, r["label"].capitalize(), r["file"]),  # optional
        "label": r["label"],                                   # optional
    }
    for r in records_filtered]

    print(descriptions)

    base_dataset = BaseConcatDataset([BaseDataset(raw, pd.Series(d)) for raw, d in zip(parts, descriptions)])
    # display(base_dataset)


    windows_dataset = create_windows_from_events(
        base_dataset,
        trial_start_offset_samples=0,
        trial_stop_offset_samples=0,
        window_size_samples=window_size_samples,
        window_stride_samples=window_size_samples,
        drop_last_window=False,
        mapping={'Bgrnd': 0, 'Bgrnd_autism': 1},
        n_jobs=4
    )


    # display(windows_dataset)

    n_classes = 2
    # Extract number of chans and time steps from dataset
    n_channels = windows_dataset[0][0].shape[0]
    input_window_samples = windows_dataset[0][0].shape[1]

    print(f"len(windows_dataset) = {len(windows_dataset)}, n_classes = {n_classes}, n_channels = {n_channels}, input_window_samples = {input_window_samples}")

    base_dataset.save(path=dataset_dir, overwrite=True)
    windows_dataset.save(path=windows_dataset_dir, overwrite=True)


if __name__ == '__main__':
    main()