import json
import os
import shutil
from tqdm import tqdm

# 파일 경로 설정
_synth_dir = os.environ.get('SYNTH_DIR', '/home/woody/workspace/IO/Data/Synth')
claude_file = os.environ.get('CLAUDE_FILE', os.path.join(_synth_dir, 'synth_gemini_step4_10_labeled_claude.json'))
gemini_file = os.environ.get('GEMINI_FILE', os.path.join(_synth_dir, 'synth_gemini_step4_10_labeled_gemini.json'))
gpt_file = os.environ.get('GPT_FILE', os.path.join(_synth_dir, 'synth_gemini_step4_10_labeled_gpt.json'))
output_file = os.environ.get('OUTPUT_FILE', os.path.join(_synth_dir, 'synth_gemini_step4_10_labeled_sum.json'))

# 각 파일을 로드하여 (topic, dialogue)를 키로 하는 딕셔너리 생성
def load_labels(file_path, label_key):
    labels = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                key = (entry['topic'], entry['dialogue'])
                labels[key] = entry.get(label_key, "unknown")
    else:
        print(f"파일을 찾을 수 없습니다: {file_path}")
    return labels

# 'theme' 필드 로드 (우선 Claude 파일에서 로드)
def load_theme(file_path):
    themes = {}
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for entry in data:
                key = (entry['topic'], entry['dialogue'])
                themes[key] = entry.get("theme", "unknown")
    else:
        print(f"파일을 찾을 수 없습니다: {file_path}")
    return themes

# 각 모델의 레이블 로드
claude_labels = load_labels(claude_file, "claude-3-haiku-20240307")
gemini_labels = load_labels(gemini_file, "gemini-1.5-flash")
gpt_labels = load_labels(gpt_file, "gpt-4o-mini")

# 'theme' 필드 로드 (Claude 파일에서 로드)
claude_themes = load_theme(claude_file)

# 원본 파일의 순서를 유지하기 위해 Claude 파일의 데이터를 기준으로 병합
if os.path.exists(claude_file):
    with open(claude_file, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
else:
    raise FileNotFoundError(f"원본 파일을 찾을 수 없습니다: {claude_file}")

# 병합된 데이터를 저장할 리스트 초기화
merged_data = []

print("레이블 병합을 시작합니다...")

for entry in tqdm(original_data, desc="레이블 병합 중"):
    topic = entry['topic']
    dialogue = entry['dialogue']
    key = (topic, dialogue)
    
    merged_entry = {
        "topic": topic,
        "dialogue": dialogue
    }
    
    # 각 모델의 레이블 추가
    merged_entry["claude-3-haiku-20240307"] = claude_labels.get(key, "unknown")
    merged_entry["gemini-1.5-flash"] = gemini_labels.get(key, "unknown")
    merged_entry["gpt-4o-mini"] = gpt_labels.get(key, "unknown")
    
    # 'theme' 필드 추가 (Claude의 'theme' 사용)
    merged_entry["theme"] = claude_themes.get(key, "unknown")
    
    # 최종 레이블 결정
    labels = [
        merged_entry["theme"],
        merged_entry["claude-3-haiku-20240307"],
        merged_entry["gemini-1.5-flash"],
        merged_entry["gpt-4o-mini"]
    ]
    
    # 레이블 빈도수 계산
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # 3가지 이상 일치하는 레이블 찾기
    final_label = "unvalid"
    for label, count in label_counts.items():
        if count >= 3 and label in ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]:
            final_label = label
            break
    
    merged_entry["label"] = final_label
    
    merged_data.append(merged_entry)

# 병합된 데이터를 JSON 파일로 저장
with open(output_file, 'w', encoding='utf-8') as f_out:
    json.dump(merged_data, f_out, ensure_ascii=False, indent=4)

print(f"레이블 병합이 완료되었습니다. 결과 파일: {output_file}")
