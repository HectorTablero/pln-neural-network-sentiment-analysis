from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Dense, Dropout, LSTM, Bidirectional,
    Embedding, Input
)
from tensorflow.keras.models import Sequential
import tensorflow as tf
from corpus import Document, PreprocessingPipeline
import sys
from pathlib import Path
import numpy as np
import polars as pl
import gensim.downloader as api
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Añadir directorio padre al path para importar corpus
sys.path.insert(0, str(Path(__file__).parent.parent / 'pln_p3_7461_01'))


def load_documents_from_csv(data_dir: Path, split: str = 'train'):
    """
    Carga documentos desde CSV.

    Args:
        data_dir: Directorio de datos
        split: 'train', 'validation' o 'test'

    Returns:
        Tupla (documentos, etiquetas)
    """
    processed_dir = data_dir / 'processed_data'
    file_path = processed_dir / f"{split}_set.csv"

    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")

    df = pl.read_csv(file_path, schema_overrides={'user': pl.Utf8})

    documents = []
    labels = []

    for row in df.iter_rows(named=True):
        doc = Document(
            doc_id=row['doc_id'],
            raw_text=row['raw_text'],
            rating=row['rating'],
            metadata={
                'game_id': row['game_id'],
                'user': row['user'],
            }
        )
        doc.label = row['label']
        documents.append(doc)
        labels.append(row['label'])

    return documents, np.array(labels)


def load_word2vec_model(model_name: str = 'glove-wiki-gigaword-100'):
    print(f"Cargando modelo Word2Vec: {model_name}...")
    model = api.load(model_name)
    print("Modelo cargado")
    return model


def text_to_embedding_average(text: str, w2v_model, preprocessing_pipeline) -> np.ndarray:
    processed_text = preprocessing_pipeline.process(text)
    words = processed_text.split()

    embeddings = []
    for word in words:
        if word in w2v_model.key_to_index:
            embeddings.append(w2v_model[word])

    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)


def text_to_word_indices(text: str, w2v_model, preprocessing_pipeline, word_to_idx: dict) -> list:
    processed_text = preprocessing_pipeline.process(text)
    words = processed_text.split()

    indices = []
    for word in words:
        if word in word_to_idx:
            indices.append(word_to_idx[word])

    return indices


def create_embedding_matrix(w2v_model, word_to_idx: dict) -> np.ndarray:
    vocab_size = len(word_to_idx) + 1  # +1 para padding (índice 0)
    embedding_dim = w2v_model.vector_size
    embedding_matrix = np.zeros((vocab_size, embedding_dim))

    for word, idx in word_to_idx.items():
        if word in w2v_model.key_to_index:
            embedding_matrix[idx] = w2v_model[word]

    return embedding_matrix


def build_vocabulary(documents: list, w2v_model, preprocessing_pipeline) -> dict:
    word_to_idx = {}
    idx = 1

    for doc in documents:
        processed_text = preprocessing_pipeline.process(doc.raw_text)
        words = processed_text.split()

        for word in words:
            if word in w2v_model.key_to_index and word not in word_to_idx:
                word_to_idx[word] = idx
                idx += 1

    return word_to_idx


def build_fnn_model(input_dim: int, num_classes: int) -> Sequential:
    """
    Construye un modelo Feedforward Neural Network (FNN).

    Args:
        input_dim: Dimensión de entrada (tamaño del embedding)
        num_classes: Número de clases de salida

    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_rnn_model(vocab_size: int, embedding_dim: int, embedding_matrix: np.ndarray,
                    max_length: int, num_classes: int) -> Sequential:
    """
    Construye un modelo RNN con LSTM bidireccional.

    Args:
        vocab_size: Tamaño del vocabulario
        embedding_dim: Dimensión de los embeddings
        embedding_matrix: Matriz de embeddings preentrenados
        max_length: Longitud máxima de secuencia
        num_classes: Número de clases de salida

    Returns:
        Modelo Keras compilado
    """
    model = Sequential([
        Embedding(
            vocab_size, embedding_dim,
            weights=[embedding_matrix],
            input_length=max_length,
            trainable=False  # Mantener embeddings fijos
        ),
        Bidirectional(LSTM(64, return_sequences=True)),
        Bidirectional(LSTM(32)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def evaluate_model(y_true, y_pred, label_encoder):
    # Decodificar etiquetas
    y_true_labels = label_encoder.inverse_transform(y_true)
    y_pred_labels = label_encoder.inverse_transform(y_pred)

    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }

    return metrics, y_true_labels, y_pred_labels


def print_evaluation_report(metrics: dict, y_true_labels, y_pred_labels, model_name: str):
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
    MAX_SEQUENCE_LENGTH = 200
    EPOCHS = 10
    BATCH_SIZE = 32

    print("\n[1] Cargando conjuntos de datos...")
    print("-" * 60)

    train_docs, y_train_labels = load_documents_from_csv(DATA_DIR, 'train')
    val_docs, y_val_labels = load_documents_from_csv(DATA_DIR, 'validation')
    test_docs, y_test_labels = load_documents_from_csv(DATA_DIR, 'test')

    print(f"  Train:      {len(train_docs)} documentos")
    print(f"  Validation: {len(val_docs)} documentos")
    print(f"  Test:       {len(test_docs)} documentos")

    # Codificar etiquetas
    label_encoder = LabelEncoder()
    label_encoder.fit(['negative', 'neutral', 'positive'])

    y_train = label_encoder.transform(y_train_labels)
    y_val = label_encoder.transform(y_val_labels)
    y_test = label_encoder.transform(y_test_labels)

    num_classes = len(label_encoder.classes_)
    print(f"  Clases: {list(label_encoder.classes_)}")

    print("\n[2] Cargando modelo Word2Vec preentrenado...")
    print("-" * 60)

    # Usar modelo más pequeño para agilizar la ejecución
    w2v_model = load_word2vec_model('glove-wiki-gigaword-100')
    embedding_dim = w2v_model.vector_size

    # =========================================================================
    # 3. PREPROCESAMIENTO Y GENERACIÓN DE EMBEDDINGS
    # =========================================================================
    print("\n[3] Preprocesando textos y generando embeddings...")
    print("-" * 60)

    preprocessing = PreprocessingPipeline([
        'remove_html',
        'remove_urls',
        'expand_contractions',
        'lowercase',
        'remove_numbers',
        'remove_punctuation',
        'remove_extra_whitespace'
    ])

    # 3.1. Embeddings promediados para FNN
    print("\n  [3.1] Generando embeddings promediados para FNN...")

    X_train_avg = np.array([
        text_to_embedding_average(doc.raw_text, w2v_model, preprocessing)
        for doc in train_docs
    ])
    X_val_avg = np.array([
        text_to_embedding_average(doc.raw_text, w2v_model, preprocessing)
        for doc in val_docs
    ])
    X_test_avg = np.array([
        text_to_embedding_average(doc.raw_text, w2v_model, preprocessing)
        for doc in test_docs
    ])

    print(f"    Train: {X_train_avg.shape}")
    print(f"    Val:   {X_val_avg.shape}")
    print(f"    Test:  {X_test_avg.shape}")

    # 3.2. Secuencias para RNN
    print("\n  [3.2] Generando secuencias para RNN...")

    word_to_idx = build_vocabulary(train_docs, w2v_model, preprocessing)
    print(f"    Vocabulario: {len(word_to_idx)} palabras")

    X_train_seq = [
        text_to_word_indices(doc.raw_text, w2v_model,
                             preprocessing, word_to_idx)
        for doc in train_docs
    ]
    X_val_seq = [
        text_to_word_indices(doc.raw_text, w2v_model,
                             preprocessing, word_to_idx)
        for doc in val_docs
    ]
    X_test_seq = [
        text_to_word_indices(doc.raw_text, w2v_model,
                             preprocessing, word_to_idx)
        for doc in test_docs
    ]

    # Padding
    X_train_pad = pad_sequences(
        X_train_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    X_val_pad = pad_sequences(
        X_val_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    X_test_pad = pad_sequences(
        X_test_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

    print(f"    Train padded: {X_train_pad.shape}")
    print(f"    Val padded:   {X_val_pad.shape}")
    print(f"    Test padded:  {X_test_pad.shape}")

    # Matriz de embeddings para RNN
    embedding_matrix = create_embedding_matrix(w2v_model, word_to_idx)
    vocab_size = len(word_to_idx) + 1
    print(f"    Matriz embeddings: {embedding_matrix.shape}")

    print("\n[4] Entrenando modelo FNN...")
    print("-" * 60)

    fnn_model = build_fnn_model(embedding_dim, num_classes)
    fnn_model.summary()

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    history_fnn = fnn_model.fit(
        X_train_avg, y_train,
        validation_data=(X_val_avg, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluar FNN
    y_pred_fnn = np.argmax(fnn_model.predict(X_test_avg), axis=1)
    metrics_fnn, y_true_labels_fnn, y_pred_labels_fnn = evaluate_model(
        y_test, y_pred_fnn, label_encoder
    )
    print_evaluation_report(metrics_fnn, y_true_labels_fnn,
                            y_pred_labels_fnn, "FNN (Feedforward Neural Network)")

    print("\n[5] Entrenando modelo RNN (LSTM Bidireccional)...")
    print("-" * 60)

    rnn_model = build_rnn_model(
        vocab_size, embedding_dim, embedding_matrix,
        MAX_SEQUENCE_LENGTH, num_classes
    )
    rnn_model.summary()

    history_rnn = rnn_model.fit(
        X_train_pad, y_train,
        validation_data=(X_val_pad, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )

    # Evaluar RNN
    y_pred_rnn = np.argmax(rnn_model.predict(X_test_pad), axis=1)
    metrics_rnn, y_true_labels_rnn, y_pred_labels_rnn = evaluate_model(
        y_test, y_pred_rnn, label_encoder
    )
    print_evaluation_report(metrics_rnn, y_true_labels_rnn,
                            y_pred_labels_rnn, "RNN (LSTM Bidireccional)")

    print("\n" + "="*80)
    print("COMPARACIÓN DE MODELOS")
    print("="*80)

    print(f"\n{'Modelo':<30s} {'Accuracy':<12s} {'Precision':<12s} {'Recall':<12s} {'F1-Score':<12s}")
    print("-" * 80)
    print(f"{'FNN':<30s} {metrics_fnn['accuracy']:<12.4f} {metrics_fnn['precision']:<12.4f} "
          f"{metrics_fnn['recall']:<12.4f} {metrics_fnn['f1']:<12.4f}")
    print(f"{'RNN (Bi-LSTM)':<30s} {metrics_rnn['accuracy']:<12.4f} {metrics_rnn['precision']:<12.4f} "
          f"{metrics_rnn['recall']:<12.4f} {metrics_rnn['f1']:<12.4f}")

    results_dir = Path(__file__).parent / 'results'
    results_dir.mkdir(exist_ok=True)

    results_file = results_dir / 'e1_neural_networks_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("1. CONFIGURACIÓN\n")
        f.write("-"*40 + "\n")
        f.write(f"  Modelo Word2Vec: glove-wiki-gigaword-100\n")
        f.write(f"  Dimensión embeddings: {embedding_dim}\n")
        f.write(f"  Longitud máx. secuencia: {MAX_SEQUENCE_LENGTH}\n")
        f.write(f"  Epochs: {EPOCHS}\n")
        f.write(f"  Batch size: {BATCH_SIZE}\n\n")

        f.write("2. RESULTADOS FNN\n")
        f.write("-"*40 + "\n")
        f.write(f"  Accuracy:  {metrics_fnn['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics_fnn['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics_fnn['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics_fnn['f1']:.4f}\n\n")
        f.write(classification_report(y_true_labels_fnn, y_pred_labels_fnn))
        f.write("\n\n")

        f.write("3. RESULTADOS RNN (Bi-LSTM)\n")
        f.write("-"*40 + "\n")
        f.write(f"  Accuracy:  {metrics_rnn['accuracy']:.4f}\n")
        f.write(f"  Precision: {metrics_rnn['precision']:.4f}\n")
        f.write(f"  Recall:    {metrics_rnn['recall']:.4f}\n")
        f.write(f"  F1-Score:  {metrics_rnn['f1']:.4f}\n\n")
        f.write(classification_report(y_true_labels_rnn, y_pred_labels_rnn))

    print(f"\n Resultados guardados en: {results_file}")


if __name__ == "__main__":
    main()
