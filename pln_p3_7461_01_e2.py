from tqdm import tqdm
from torch.optim import AdamW
from transformers import (
    BertTokenizer,
    BertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from torch.utils.data import Dataset, DataLoader
import torch
import sys
from pathlib import Path
import numpy as np
import polars as pl
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Añadir directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'pln_p3_7461_01'))


def get_device():
    """Obtiene el dispositivo disponible (GPU/CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Usando GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Usando CPU")
    return device


class ReviewDataset(Dataset):
    """Dataset personalizado para reseñas de BGG."""

    def __init__(self, texts: list, labels: np.ndarray, tokenizer, max_length: int = 128):
        """
        Inicializa el dataset.

        Args:
            texts: Lista de textos
            labels: Array de etiquetas codificadas
            tokenizer: Tokenizador BERT
            max_length: Longitud máxima de tokens
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


def load_documents_from_csv(data_dir: Path, split: str = 'train'):

    processed_dir = data_dir / 'processed_data'
    file_path = processed_dir / f"{split}_set.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    df = pl.read_csv(file_path, schema_overrides={'user': pl.Utf8})

    texts = df['raw_text'].to_list()
    labels = df['label'].to_list()

    return texts, np.array(labels)


def train_epoch(model, data_loader, optimizer, scheduler, device):
    """
    Entrena el modelo por una época.

    Args:
        model: Modelo BERT
        data_loader: DataLoader de entrenamiento
        optimizer: Optimizador
        scheduler: Scheduler de learning rate
        device: Dispositivo (GPU/CPU)

    Returns:
        Pérdida promedio de la época
    """
    model.train()
    total_loss = 0

    progress_bar = tqdm(data_loader, desc="Training", leave=False)

    for batch in progress_bar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        progress_bar.set_postfix({'loss': loss.item()})

    return total_loss / len(data_loader)


def evaluate(model, data_loader, device):
    """
    Evalúa el modelo.

    Args:
        model: Modelo BERT
        data_loader: DataLoader de evaluación
        device: Dispositivo (GPU/CPU)

    Returns:
        Tupla (pérdida, predicciones, etiquetas verdaderas)
    """
    model.eval()
    total_loss = 0
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_loss += outputs.loss.item()

            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)

            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return total_loss / len(data_loader), np.array(predictions), np.array(true_labels)


def compute_metrics(y_true, y_pred):
    """
    Calcula métricas de clasificación.

    Args:
        y_true: Etiquetas verdaderas
        y_pred: Predicciones

    Returns:
        Diccionario con métricas
    """
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }


def print_evaluation_report(metrics: dict, y_true_labels, y_pred_labels, model_name: str):
    """
    Imprime un reporte completo de evaluación.

    Args:
        metrics: Diccionario con métricas
        y_true_labels: Etiquetas verdaderas
        y_pred_labels: Predicciones
        model_name: Nombre del modelo
    """
    print(f"\n{'='*60}")
    print(f"EVALUACIÓN: {model_name}")
    print('='*60)

    print(f"\nMétricas generales:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1-Score:  {metrics['f1']:.4f}")

    print(f"\nReporte de clasificación:")
    print(classification_report(y_true_labels, y_pred_labels))

    print(f"\nMatriz de confusión:")
    cm = confusion_matrix(y_true_labels, y_pred_labels)
    labels = sorted(set(y_true_labels))
    print(f"{'':12s}", end='')
    for label in labels:
        print(f"{label[:8]:>10s}", end='')
    print()
    for i, label in enumerate(labels):
        print(f"{label[:10]:12s}", end='')
        for j in range(len(labels)):
            print(f"{cm[i][j]:10d}", end='')
        print()


def main():

    # Configuración
    DATA_DIR = Path(__file__).parent.parent / 'pln_p3_7461_01' / 'data'
    MODEL_NAME = 'bert-base-uncased'
    MAX_LENGTH = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    LEARNING_RATE = 2e-5

    print("\n[1] Configurando dispositivo...")
    print("-" * 60)
    device = get_device()

    print("\n[2] Cargando conjuntos de datos...")
    print("-" * 60)

    train_texts, y_train_labels = load_documents_from_csv(DATA_DIR, 'train')
    val_texts, y_val_labels = load_documents_from_csv(DATA_DIR, 'validation')
    test_texts, y_test_labels = load_documents_from_csv(DATA_DIR, 'test')

    print(f"Train:      {len(train_texts)} documentos")
    print(f"Validation: {len(val_texts)} documentos")
    print(f"Test:       {len(test_texts)} documentos")

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    label_encoder.fit(['negative', 'neutral', 'positive'])

    y_train = label_encoder.transform(y_train_labels)
    y_val = label_encoder.transform(y_val_labels)
    y_test = label_encoder.transform(y_test_labels)

    num_classes = len(label_encoder.classes_)
    print(f"Clases: {list(label_encoder.classes_)}")

    print("\n[3] Cargando tokenizador y modelo BERT...")
    print("-" * 60)

    print(f"  Cargando {MODEL_NAME}...")
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
    model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_classes
    )
    model.to(device)

    print(f"Tokenizador cargado")
    print(f"Modelo cargado con {num_classes} clases de salida")

    print("\n[4] Creando datasets y dataloaders...")
    print("-" * 60)

    train_dataset = ReviewDataset(train_texts, y_train, tokenizer, MAX_LENGTH)
    val_dataset = ReviewDataset(val_texts, y_val, tokenizer, MAX_LENGTH)
    test_dataset = ReviewDataset(test_texts, y_test, tokenizer, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    print(f"Train DataLoader: {len(train_loader)} batches")
    print(f"Val DataLoader:   {len(val_loader)} batches")
    print(f"Test DataLoader:  {len(test_loader)} batches")

    print("\n[5] Configurando optimizador y scheduler...")
    print("-" * 60)

    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )

    print(f"Optimizador: AdamW (lr={LEARNING_RATE})")
    print(f"Total steps: {total_steps}")

    print("\n[6] Entrenando modelo BERT...")
    print("-" * 60)

    best_val_f1 = 0
    best_model_state = None

    for epoch in range(EPOCHS):
        print(f"\nÉpoca {epoch + 1}/{EPOCHS}")
        print("-" * 40)

        # Entrenar
        train_loss = train_epoch(
            model, train_loader, optimizer, scheduler, device)
        print(f"  Train Loss: {train_loss:.4f}")

        # Evaluar en validación
        val_loss, val_preds, val_true = evaluate(model, val_loader, device)
        val_metrics = compute_metrics(val_true, val_preds)

        print(f"Val Loss:   {val_loss:.4f}")
        print(f"Val F1:     {val_metrics['f1']:.4f}")
        print(f"Val Acc:    {val_metrics['accuracy']:.4f}")

        # Guardar mejor modelo
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            best_model_state = model.state_dict().copy()
            print(f"  Nuevo mejor modelo guardado!")

    # Cargar mejor modelo
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print("\n[7] Evaluando en conjunto de test...")
    print("-" * 60)

    _, test_preds, test_true = evaluate(model, test_loader, device)
    test_metrics = compute_metrics(test_true, test_preds)

    # Decodificar etiquetas para el reporte
    test_true_labels = label_encoder.inverse_transform(test_true)
    test_pred_labels = label_encoder.inverse_transform(test_preds)

    print_evaluation_report(
        test_metrics, test_true_labels, test_pred_labels,
        "BERT Fine-tuned")

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / 'e2_bert_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("1. CONFIGURACIÓN\n")
        f.write("-"*40 + "\n")
        f.write(f"  Modelo base: {MODEL_NAME}\n")
        f.write(f"  Max length: {MAX_LENGTH}\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n")
        f.write(f"  Epochs: {EPOCHS}\n")
        f.write(f"  Learning rate: {LEARNING_RATE}\n\n")

        f.write("2. RESULTADOS EN TEST\n")
        f.write("-"*40 + "\n")
        f.write(f"  Accuracy:  {test_metrics['accuracy']:.4f}\n")
        f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
        f.write(f"  Recall:    {test_metrics['recall']:.4f}\n")
        f.write(f"  F1-Score:  {test_metrics['f1']:.4f}\n\n")

        f.write("3. REPORTE DE CLASIFICACIÓN\n")
        f.write("-"*40 + "\n")
        f.write(classification_report(test_true_labels, test_pred_labels))
        f.write("\n")

        f.write("4. MATRIZ DE CONFUSIÓN\n")
        f.write("-"*40 + "\n")
        cm = confusion_matrix(test_true_labels, test_pred_labels)
        labels = sorted(set(test_true_labels))
        f.write(f"{'':12s}")
        for label in labels:
            f.write(f"{label[:8]:>10s}")
        f.write("\n")
        for i, label in enumerate(labels):
            f.write(f"{label[:10]:12s}")
            for j in range(len(labels)):
                f.write(f"{cm[i][j]:10d}")
            f.write("\n")

    print(f"\nResultados guardados en: {results_file}")

    # Guardar modelo
    model_path = results_dir / 'bert_model'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Modelo guardado en: {model_path}")


if __name__ == "__main__":
    main()
