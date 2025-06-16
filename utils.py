import numpy as np

def generate_similarity_map_html(top_pairs, source_sents, compared_sents, num_to_show):
    num_colors = min(len(top_pairs), int(num_to_show))
    
    gradient_colors = []
    for i in range(num_colors):
        hue = 0 + (i / max(1, num_colors - 1)) * 40 
        gradient_colors.append(f"hsl({hue}, 100%, 50%)")

    source_highlights = {}
    compared_highlights = {}
    for i, pair in enumerate(top_pairs[:num_colors]):
        color = gradient_colors[i]
        src_idx = pair['source_idx']
        cmp_idx = pair['compared_idx']
        source_highlights.setdefault(src_idx, color)
        compared_highlights.setdefault(cmp_idx, color)
   
    def create_html_block(sentences, highlights):
        html_parts = []
        for i, sent in enumerate(sentences):
            safe_sent = sent.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            if i in highlights:
                style = f'background-color: {highlights[i]}; color: white; padding: 2px 4px; border-radius: 4px;'
                html_parts.append(f'<mark style="{style}">{safe_sent}</mark>')
            else:
                html_parts.append(safe_sent)
        return " ".join(html_parts)

    source_html = create_html_block(source_sents, source_highlights)
    compared_html = create_html_block(compared_sents, compared_highlights)
    header = "<h4>Подсветка наиболее похожих предложений (от красного к оранжевому):</h4>"
    styles = "display:flex; justify-content: space-between; gap: 20px;"
    source_block = f'<div style="flex: 1; border: 1px solid #ddd; padding: 10px; border-radius: 5px;"><b>Исходный текст:</b><p>{source_html}</p></div>'
    compared_block = f'<div style="flex: 1; border: 1px solid #ddd; padding: 10px; border-radius: 5px;"><b>Сравниваемый текст:</b><p>{compared_html}</p></div>'
    
    return f"{header}<div style='{styles}'>{source_block}{compared_block}</div>"

def find_top_sentence_pairs(similarity_matrix, top_k):
    flat_scores = similarity_matrix.flatten()
    num_candidates = min(int(top_k) * 5, len(flat_scores))
    if num_candidates == 0:
        return []
    top_indices_flat = np.argpartition(flat_scores, -num_candidates)[-num_candidates:]
    top_scores = flat_scores[top_indices_flat]
    sorted_top_indices = top_indices_flat[np.argsort(-top_scores)]
    rows, cols = np.unravel_index(sorted_top_indices, similarity_matrix.shape)
    
    unique_pairs = []
    seen_src_idx = set()
    seen_cmp_idx = set()
    
    for r, c in zip(rows, cols):
        if r not in seen_src_idx and c not in seen_cmp_idx:
            unique_pairs.append({
                'source_idx': r,
                'compared_idx': c,
                'score': similarity_matrix[r, c]
            })
            seen_src_idx.add(r)
            seen_cmp_idx.add(c)
        if len(unique_pairs) >= int(top_k):
            break
            
    return unique_pairs

def load_source_text_from_file(file_obj):
    if not file_obj:
        return ""
    try:
        with open(file_obj.name, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        return f"Не удалось прочитать файл: {e}" 