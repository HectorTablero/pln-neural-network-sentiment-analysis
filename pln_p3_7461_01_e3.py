import sys
from pathlib import Path
import polars as pl
import json
from typing import Optional

# Añadir directorio padre al path
sys.path.insert(0, str(Path(__file__).parent.parent / 'pln_p3_7461_01'))


def load_corpus(data_dir: Path) -> pl.DataFrame:
    processed_dir = data_dir / 'processed_data'

    # Cargar todos los splits
    dfs = []
    for split in ['train', 'validation', 'test']:
        file_path = processed_dir / f"{split}_set.csv"
        if file_path.exists():
            df = pl.read_csv(file_path, schema_overrides={'user': pl.Utf8})
            dfs.append(df)

    return pl.concat(dfs)


def get_reviews_by_game(df: pl.DataFrame, game_id: int, max_reviews: int = 20) -> pl.DataFrame:
    reviews = df.filter(pl.col('game_id') == game_id)

    # Balancear por etiquetas si hay muchas reseñas
    if len(reviews) > max_reviews:
        # Intentar obtener una muestra balanceada
        positive = reviews.filter(
            pl.col('label') == 'positive').head(max_reviews // 3)
        negative = reviews.filter(
            pl.col('label') == 'negative').head(max_reviews // 3)
        neutral = reviews.filter(
            pl.col('label') == 'neutral').head(max_reviews // 3)
        reviews = pl.concat([positive, negative, neutral])

    return reviews


def get_available_games(df: pl.DataFrame, min_reviews: int = 10) -> pl.DataFrame:
    return (
        df.group_by('game_id')
        .agg(pl.count().alias('num_reviews'))
        .filter(pl.col('num_reviews') >= min_reviews)
        .sort('num_reviews', descending=True)
    )


def build_opinion_summary_prompt(reviews: pl.DataFrame, game_id: int) -> str:
    """
    Construye un prompt estructurado para generar un resumen de opinión.

    Args:
        reviews: DataFrame con las reseñas del juego
        game_id: ID del juego

    Returns:
        Prompt formateado para el LLM
    """
    # Estadísticas de las reseñas
    total_reviews = len(reviews)
    avg_rating = reviews['rating'].mean()

    # Conteo por sentimiento
    sentiment_counts = reviews.group_by('label').agg(pl.count().alias('count'))
    sentiments = {row['label']: row['count']
                  for row in sentiment_counts.iter_rows(named=True)}

    # Construir sección de reseñas
    reviews_text = ""
    for i, row in enumerate(reviews.iter_rows(named=True), 1):
        reviews_text += f"\n--- Reseña {i} (Rating: {row['rating']}, Sentimiento: {row['label']}) ---\n"
        reviews_text += f"{row['raw_text'][:500]}{'...' if len(row['raw_text']) > 500 else ''}\n"

    prompt = f"""Eres un experto analista de opiniones de juegos de mesa. A continuación se te proporcionan {total_reviews} reseñas de usuarios sobre un juego de mesa (Game ID: {game_id}).

ESTADÍSTICAS DEL JUEGO:
- Total de reseñas analizadas: {total_reviews}
- Rating promedio: {avg_rating:.2f}/10
- Distribución de sentimientos:
  * Positivas: {sentiments.get('positive', 0)}
  * Negativas: {sentiments.get('negative', 0)}
  * Neutras: {sentiments.get('neutral', 0)}

RESEÑAS:
{reviews_text}

INSTRUCCIONES:
Genera un resumen de opinión agregada que incluya:

1. **OPINIÓN GENERAL**: Un párrafo breve describiendo la percepción general del juego.

2. **PUNTOS FUERTES**: Lista de los aspectos más valorados positivamente por los usuarios, con evidencias de las reseñas.

3. **PUNTOS DÉBILES**: Lista de las principales críticas o aspectos negativos mencionados, con evidencias.

4. **OPINIONES ENCONTRADAS**: Si existen, menciona aspectos sobre los que hay opiniones divididas.

5. **RECOMENDACIÓN**: A qué tipo de jugador se recomienda este juego y a cuál no.

Responde de forma estructurada y concisa, basándote únicamente en la información de las reseñas proporcionadas.
"""

    return prompt


def build_simple_prompt(reviews: pl.DataFrame, game_id: int) -> str:
    """
    Construye un prompt más simple y directo.

    Args:
        reviews: DataFrame con las reseñas
        game_id: ID del juego

    Returns:
        Prompt simplificado
    """
    # Concatenar reseñas con contexto
    reviews_list = []
    for row in reviews.iter_rows(named=True):
        text = row['raw_text'][:300] + \
            ('...' if len(row['raw_text']) > 300 else '')
        reviews_list.append(f"[Rating: {row['rating']}/10] {text}")

    reviews_text = "\n\n".join(reviews_list)

    prompt = f"""Analiza las siguientes {len(reviews)} reseñas de un juego de mesa y genera un resumen conciso de la opinión general.

RESEÑAS:
{reviews_text}

Proporciona:
1. Resumen de opinión general (2-3 oraciones)
2. Principales aspectos positivos (lista)
3. Principales aspectos negativos (lista)
4. Conclusión y recomendación (1-2 oraciones)
"""

    return prompt


def generate_summary_with_ollama(prompt: str, model: str = "qwen3:4b") -> Optional[str]:
    """
    Genera un resumen usando Ollama (modelo local).

    Args:
        prompt: Prompt para el LLM
        model: Nombre del modelo Ollama

    Returns:
        Resumen generado o None si falla
    """
    try:
        import ollama

        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response['message']['content']
    except ImportError:
        print("  ⚠ Ollama no está instalado. Instálalo con: pip install ollama")
        return None
    except Exception as e:
        print(f"  ⚠ Error con Ollama: {e}")
        return None


def main():
    # Configuración
    DATA_DIR = Path(__file__).parent.parent / 'pln_p3_7461_01' / 'data'
    RESULTS_DIR = Path(__file__).parent / 'results'
    RESULTS_DIR.mkdir(exist_ok=True)

    print("\n[1] Cargando corpus de reseñas...")
    print("-" * 60)

    df = load_corpus(DATA_DIR)
    print(f"  Total de reseñas: {len(df)}")

    # Obtener juegos disponibles
    available_games = get_available_games(df, min_reviews=15)
    print(f"  Juegos con ≥15 reseñas: {len(available_games)}")

    print("\n  Top 10 juegos por número de reseñas:")
    for i, row in enumerate(available_games.head(10).iter_rows(named=True), 1):
        print(
            f"    {i}. Game ID: {row['game_id']:3d} - {row['num_reviews']} reseñas")

    print("\n[2] Seleccionando juego para análisis...")
    print("-" * 60)

    # Seleccionar el juego con más reseñas
    top_game = available_games.head(1)
    game_id = top_game['game_id'][0]

    reviews = get_reviews_by_game(df, game_id, max_reviews=15)
    print(f"  Game ID seleccionado: {game_id}")
    print(f"  Reseñas para análisis: {len(reviews)}")

    # Mostrar distribución de sentimientos
    sentiment_dist = reviews.group_by('label').agg(pl.count().alias('count'))
    print(f"\n  Distribución de sentimientos:")
    for row in sentiment_dist.iter_rows(named=True):
        print(f"    {row['label']}: {row['count']}")

    print("\n[3] Construyendo prompts...")
    print("-" * 60)

    # Prompt detallado
    detailed_prompt = build_opinion_summary_prompt(reviews, game_id)

    # Prompt simple
    simple_prompt = build_simple_prompt(reviews, game_id)

    print(f"  Prompt detallado: {len(detailed_prompt)} caracteres")
    print(f"  Prompt simple: {len(simple_prompt)} caracteres")

    print("\n[4] Intentando generar resumen automáticamente...")
    print("-" * 60)

    summary = None
    method_used = None

    # Intentar con Ollama primero (local)
    print("  Intentando con Ollama (local)...")
    summary = generate_summary_with_ollama(simple_prompt)
    if summary:
        method_used = "Ollama (local)"
        print("  Resumen generado con Ollama")

    print("\n[5] Resultados...")
    print("-" * 60)

    if summary:
        print(summary)

    # Guardar resultados
    results_file = RESULTS_DIR / 'e3_opinion_summary_results.txt'
    with open(results_file, 'w', encoding='utf-8') as f:
        f.write("1. CONFIGURACIÓN\n")
        f.write("-"*40 + "\n")
        f.write(f"  Game ID analizado: {game_id}\n")
        f.write(f"  Número de reseñas: {len(reviews)}\n")
        f.write(f"  Método de prompting: {method_used}\n\n")

        f.write("2. DISTRIBUCIÓN DE SENTIMIENTOS\n")
        f.write("-"*40 + "\n")
        for row in sentiment_dist.iter_rows(named=True):
            f.write(f"  {row['label']}: {row['count']}\n")
        f.write("\n")

        f.write("3. PROMPT UTILIZADO (Simple)\n")
        f.write("-"*40 + "\n")
        f.write(simple_prompt)
        f.write("\n\n")

        f.write("4. PROMPT UTILIZADO (Detallado)\n")
        f.write("-"*40 + "\n")
        f.write(detailed_prompt)
        f.write("\n\n")

        f.write("5. RESUMEN GENERADO\n")
        f.write("-"*40 + "\n")
        f.write(summary if summary else "[PENDIENTE]")
        f.write("\n")

    print(f"\nResultados guardados en: {results_file}")


if __name__ == "__main__":
    main()
