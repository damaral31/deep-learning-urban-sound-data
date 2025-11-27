import soundata
urbansound8k = soundata.initialize('urbansound8k', data_home="C:\\Users\\diogo\\OneDrive\\Documents\\UrbanSound8K")  # get the urbansound8k dataset
urbansound8k.download(force_overwrite=True)  # download orchset
urbansound8k.validate()  # validate orchset 