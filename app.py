import gradio as gr
from sentence_transformers import SentenceTransformer, util, CrossEncoder
import pandas as pd
import os
import numpy as np
import nltk
import traceback
from utils import generate_similarity_map_html, find_top_sentence_pairs, load_source_text_from_file
from nltk.tokenize import PunktTokenizer

bi_encoder_path = 'fine-tuned-bi-encoder/checkpoint-6000'
cross_encoder_path = 'fine-tuned-cross-encoder/checkpoint-6000'
bi_encoder_default = 'paraphrase-multilingual-MiniLM-L12-v2'
cross_encoder_default = 'microsoft/deberta-v3-small'

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

sentence_tokenizer = PunktTokenizer()
top_k_candidates = 10

try:
    bi_encoder = SentenceTransformer(bi_encoder_path)
except Exception as e:
    bi_encoder = SentenceTransformer(bi_encoder_default)
try:
    cross_encoder = CrossEncoder(cross_encoder_path)
except Exception as e: 
    cross_encoder = CrossEncoder(cross_encoder_default)

print("Модели загружены.")


def check_text_plagiarism(source_text, files, gradio_state_dict):
    if not source_text or not files:
        empty_df = pd.DataFrame(columns=["Файл", "Быстрая оценка", "Точная оценка"])
        return empty_df, "Выберите файл из таблицы для просмотра деталей.", gradio_state_dict

    file_contents = {}
    for file in files:
        with open(file.name, 'r', encoding='utf-8') as f:
            file_contents[os.path.basename(file.name)] = f.read()
    
    source_embedding = bi_encoder.encode(source_text, convert_to_tensor=True)
    corpus_embeddings = bi_encoder.encode(list(file_contents.values()), convert_to_tensor=True)
    cos_scores = util.cos_sim(source_embedding, corpus_embeddings)[0].cpu()

    num_candidates = min(top_k_candidates, len(cos_scores))
    top_results_indices = np.argpartition(-cos_scores, range(num_candidates))[:num_candidates]

    cross_input_pairs = []
    top_files = [list(file_contents.keys())[i] for i in top_results_indices]
    for filename in top_files:
        cross_input_pairs.append([source_text, file_contents[filename]])
    
    cross_encoder_results = {}
    if cross_input_pairs:
        probabilities = cross_encoder.predict(cross_input_pairs, apply_softmax=True, show_progress_bar=False)
        if probabilities.ndim > 1:
            plagiarism_probs = probabilities[:, 1]
        else:
            plagiarism_probs = [probabilities[1]]
        
        for filename, prob in zip(top_files, plagiarism_probs):
            cross_encoder_results[filename] = prob

    all_results = []
    for i, (filename, content) in enumerate(file_contents.items()):
        quick_score = f"{cos_scores[i]:.2f}"
        exact_score = f"{cross_encoder_results.get(filename, 0.0):.2f}" if filename in top_files else "N/A"
        all_results.append([filename, quick_score, exact_score])
        
    all_results.sort(key=lambda x: float(x[2] if x[2] != "N/A" else -1), reverse=True)
    df = pd.DataFrame(all_results, columns=["Файл", "Быстрая оценка", "Точная оценка"])

    gradio_state_dict["all_results_list"] = all_results
    gradio_state_dict["source_sents"] = sentence_tokenizer.tokenize(source_text)
    gradio_state_dict["file_sents"] = {name: sentence_tokenizer.tokenize(text) for name, text in file_contents.items()}
    gradio_state_dict["source_sents_embeddings"] = bi_encoder.encode(gradio_state_dict["source_sents"], convert_to_tensor=True)
    gradio_state_dict["file_sents_embeddings"] = {name: bi_encoder.encode(sents, convert_to_tensor=True) for name, sents in gradio_state_dict["file_sents"].items()}

    return df, "Кликните на строку в таблице, чтобы увидеть карту схожести.", gradio_state_dict

def visualize_similarity(gradio_state, num_to_show, evt: gr.SelectData):
    try:
        if evt is None or evt.index is None or not gradio_state:
            return "Нет данных для визуализации. Сначала выполните проверку, затем кликните на строку в таблице."

        selected_row_index = evt.index[0]
        all_results = gradio_state.get("all_results_list", [])
            
        selected_filename = all_results[selected_row_index][0]
        
        source_sents = gradio_state.get("source_sents", [])
        compared_sents = gradio_state.get("file_sents", {}).get(selected_filename, [])
        source_embeddings = gradio_state.get("source_sents_embeddings")
        compared_embeddings = gradio_state.get("file_sents_embeddings", {}).get(selected_filename)

        if not source_sents or not compared_sents or source_embeddings is None or compared_embeddings is None:
            return "Один из текстов пуст или не удалось получить данные для визуализации. Перезапустите проверку."

        similarity_matrix = util.cos_sim(source_embeddings, compared_embeddings).cpu().numpy()
        
        if similarity_matrix.size == 0:
            return "Не удалось построить карту сходства (пустые тексты)."

        # Вызываем утилиту для поиска пар
        unique_pairs = find_top_sentence_pairs(similarity_matrix, num_to_show)
        # Вызываем утилиту для генерации HTML
        return generate_similarity_map_html(unique_pairs, source_sents, compared_sents, int(num_to_show))

    except Exception as e:
        error_text = f"Произошла внутренняя ошибка: {e}\n{traceback.format_exc()}"
        print(error_text)
        return f'<pre style="color:red; white-space: pre-wrap;">{error_text}</pre>'


with gr.Blocks() as demo:
    gr.Markdown("# Детектор плагиата в тексте")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown(
                "**Этап 1:** Загрузите файлы, система быстро оценит их схожесть с исходным текстом (Bi-Encoder). "
                f"**Этап 2:** Топ-{top_k_candidates} самых похожих файлов будут перепроверены точной моделью (Cross-Encoder)."
            )
            text_source_input = gr.Textbox(label="Исходный текст", lines=8, placeholder="Вставьте сюда текст для проверки...")
            source_file_input = gr.File(label="...или загрузите исходный файл (.txt)", file_types=['.txt'])
            
            text_files_input = gr.File(label="Загрузите файлы для сравнения (.txt)", file_count="multiple", scale=2)

            text_compare_button = gr.Button("Проверить текст на плагиат")
        
        with gr.Column(scale=3):
            text_results_output = gr.DataFrame(
                headers=["Файл", "Быстрая оценка", "Точная оценка"],
                datatype=["str", "str", "str"],
                label="Результаты проверки"
            )
            similarity_map_output = gr.HTML(label="Карта схожести предложений")
            num_pairs_slider = gr.Slider(
                minimum=3, maximum=20, value=10, step=1, 
                label="Количество пар предложений для подсветки",
                info="Управляет детализацией визуализации ниже."
            )

    # State для хранения данных между вызовами функций. Решает проблему с глобальными переменными
    gradio_state = gr.State({})

    source_file_input.upload(
        fn=load_source_text_from_file,
        inputs=[source_file_input],
        outputs=[text_source_input],
    )

    text_compare_button.click(
        fn=check_text_plagiarism,
        inputs=[text_source_input, text_files_input, gradio_state],
        outputs=[text_results_output, similarity_map_output, gradio_state]
    )
    
    text_results_output.select(
        fn=visualize_similarity,
        inputs=[gradio_state, num_pairs_slider],
        outputs=[similarity_map_output]
    )

if __name__ == "__main__":
    demo.launch()