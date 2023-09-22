import json
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

import torch
from torch import cuda
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from Dataset import Dataset
from BertClassifier import BertClassifier

tqdm.pandas()


class Classifier:
    def __init__(self, batch_size: int = 10):
        self.categories = ['Бизнес и стартапы', 'Блоги', 'Видео и фильмы', 'Дизайн', 'Еда и кулинария',
                           'Здоровье и медицина', 'Игры', 'Искусство', 'Картинки и фото', 'Криптовалюты',
                           'Маркетинг, PR, реклама', 'Мода и красота', 'Музыка', 'Новости и СМИ',
                           'Образование и познавательное', 'Политика', 'Право', 'Психология', 'Путешествия',
                           'Развлечения и юмор', 'Рукоделие', 'Софт и приложения', 'Спорт', 'Технологии', 'Финансы',
                           'Цитаты', 'Шоубиз', 'Экономика']
        self.label_mapping = {self.categories[i]: i for i in range(len(self.categories))}
        self.num_labels = len(self.categories)

        self.device = 'cuda' if cuda.is_available() else 'cpu'
        self.model = BertClassifier(self.num_labels, self.device).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('cointegrated/rubert-tiny2')
        self.batch_size = batch_size

        self.is_trained = False

        # Поменять под удобные
        self.path_to_models = "models/"
        self.path_to_reports = "reports/"

    def eval(self, df: pd.DataFrame, epoch_idx: int = -2, save_report: bool = False) -> None:
        print("Evaluating...")

        eval_dataset = Dataset(list(df['text']), list(df['label'].map(self.label_mapping)), self.tokenizer)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size)

        eval_labels = []
        preds = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(eval_dataloader), total=len(eval_dataloader),
                                desc=f"Evaluating: ", unit=" batch", position=0, leave=True):
                inputs = data
                true_labels = data['labels']
                eval_labels = eval_labels + true_labels.detach().numpy().tolist()

                logits = self.model(inputs)

                logit_preds, label_preds = torch.max(logits.data, 1)
                preds = preds + label_preds.detach().cpu().numpy().tolist()

        if save_report:
            report = classification_report(eval_labels, preds, zero_division=0)
            report_to_save = classification_report(eval_labels, preds, zero_division=0, output_dict=True)

            with open(f"{self.path_to_reports}report_{epoch_idx}epoch.txt", 'w', encoding='utf-8') as f:
                json.dump(report_to_save, f, ensure_ascii=False, indent=4)

            print(report)

    def train(self, df: pd.DataFrame, num_epochs: int, learning_rate: float = 1e-5,
              max_len_sample_per_class: int = 1000, eval_size: float = 0.1, eval_per_epochs: int = 1,
              model_name: str = "BertClassifier") -> list:
        print("Training...")
        self.is_trained = True

        sampled_df = df.groupby('label').apply(lambda x: x.sample(min(len(x), max_len_sample_per_class))).droplevel(
            'label')

        train_data, eval_data = train_test_split(sampled_df[['text', 'label']], test_size=eval_size, random_state=42,
                                                 stratify=sampled_df['label'])

        train_dataset = Dataset(list(train_data['text']), list(train_data['label'].map(self.label_mapping)),
                                self.tokenizer)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        train_losses = []

        for epoch_idx in tqdm(range(1, num_epochs + 1), total=num_epochs, desc="Epochs: ", unit=" epoch", position=0, leave=True):
            train_epoch_losses = []
            eval_epoch_losses = []

            for i, data in tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"{epoch_idx} epoch: ",
                                unit=" batch", position=0, leave=True):
                inputs = data
                labels = data['labels'].to(self.device)

                optimizer.zero_grad()

                logits = self.model(inputs)

                loss = criterion(logits, labels)
                loss.backward()

                train_epoch_losses.append(loss.item())

                optimizer.step()

            train_losses.append(np.mean(train_epoch_losses))
            torch.save(torch.tensor(train_losses), f"{self.path_to_reports}train_losses_{epoch_idx}epoch.pt")

            if epoch_idx % eval_per_epochs == 0:
                self.eval(eval_data, epoch_idx - 1, save_report=True)

            # В проде поменяй на
            # self.save_state(f"{model_name}")
            self.save_state(f"{model_name}_{epoch_idx}epoch")

        return train_losses

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        print("Predicting...")
        df_dataset = Dataset(list(df['text']), None, self.tokenizer)
        df_dataloader = DataLoader(df_dataset, batch_size=self.batch_size)

        preds = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(df_dataloader), total=len(df_dataloader), desc=f"Predicting: ",
                                unit=" batch", position=0, leave=True):
                inputs = data

                logits = self.model(inputs)

                logit_preds, label_preds = torch.max(logits.data, 1)
                preds = preds + label_preds.detach().cpu().numpy().tolist()

        df['label'] = preds
        df['label'] = df['label'].map({v: k for k, v in self.label_mapping.items()})

        return df

    def save_state(self, model_name: str = "BertClassifier") -> None:
        torch.save(self.model.state_dict(), f"{self.path_to_models}{model_name}.pt")

    def load_state(self, model_name: str = "BertClassifier") -> None:
        self.model.load_state_dict(
            torch.load(f"{self.path_to_models}{model_name}.pt", map_location=torch.device('cpu')))
