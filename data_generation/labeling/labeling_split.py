import json
import os
from collections import Counter
from tqdm import tqdm

# 입력 파일 경로 설정
input_file = os.environ.get('INPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/synth_gemini_step4_10_labeled_sum.json')

# 입력 파일에서 데이터 로드
if not os.path.exists(input_file):
    raise FileNotFoundError(f"입력 파일을 찾을 수 없습니다: {input_file}")

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# "label" 필드의 각 항목 개수를 셀 Counter 객체 생성
label_counter = Counter()

# 각 대화 항목의 "label" 필드 값을 카운트
for dialogue in tqdm(data, desc="레이블 카운트 중"):
    label = dialogue.get("label", "unknown")  # "label" 필드가 없을 경우 "unknown"으로 설정
    label_counter[label] += 1

# 결과 출력
print("각 레이블의 개수:")
for label, count in label_counter.items():
    print(f"{label}: {count}")

# "label"이 "unvalid"가 아닌 항목들을 필터링하여 저장
valid_dialogues = [dialogue for dialogue in data if dialogue.get("label") != "unvalid"]
output_file_all = '/home/woody/workspace/Emotion-Neuron/data/step4.json'
with open(output_file_all, 'w', encoding='utf-8') as f:
    json.dump(valid_dialogues, f, ensure_ascii=False, indent=4)
print(f"'label'이 'unvalid'가 아닌 항목을 {output_file_all}에 저장했습니다.")

# # 특정 감정별로 항목을 필터링하여 저장하는 함수
# def save_dialogues_by_label(label, output_path):
#     filtered_dialogues = [dialogue for dialogue in data if dialogue.get("label") == label]
#     with open(output_path, 'w', encoding='utf-8') as f:
#         json.dump(filtered_dialogues, f, ensure_ascii=False, indent=4)
#     print(f"'label'이 '{label}'인 항목을 {output_path}에 저장했습니다.")

# # 각 감정별로 파일 저장
# emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
# for emotion in emotions:
#     output_file = f'/home/woody/workspace/Emotion-Neuron/data/step4_{emotion}.json'
#     save_dialogues_by_label(emotion, output_file)
