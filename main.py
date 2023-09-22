import pandas as pd
from Classifier import Classifier

final_dataset = pd.read_csv("final_dataset.csv")[['text', 'label']]

model = Classifier(batch_size=1)

model.train(df=final_dataset, num_epochs=20, learning_rate=1e-4, max_len_sample_per_class=200)
