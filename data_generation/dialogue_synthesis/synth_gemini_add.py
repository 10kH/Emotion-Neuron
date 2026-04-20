import json
import time
import os
from tqdm import tqdm
import google.generativeai as genai
from collections import defaultdict
import shutil
from datetime import datetime

# Gemini API 설정
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("Please set the GOOGLE_API_KEY environment variable.")

genai.configure(api_key=GOOGLE_API_KEY)

# 감정 리스트 정의
emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# 주제 파일 경로 설정
input_file = os.environ.get('INPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/fits_step4.json')
output_file = os.environ.get('OUTPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/synth_gemini_step4_10.json')

# 생성할 대화의 개수 설정 (각 주제 & 감정 쌍당)
num_conversations_per_pair = 10  # 원하는 숫자로 변경하세요

# 백업 디렉토리 설정
backup_dir = os.environ.get('BACKUP_DIR', '/home/woody/workspace/Emotion-Neuron/data/backup')
os.makedirs(backup_dir, exist_ok=True)

# 주제 데이터 로드
with open(input_file, 'r', encoding='utf-8') as f:
    topics_data = json.load(f)

# 생성된 대화 저장을 위한 리스트 초기화
generated_conversations = []

# 기존에 생성된 데이터가 있다면 로드
if os.path.exists(output_file):
    try:
        with open(output_file, 'r', encoding='utf-8') as f_out:
            generated_conversations = json.load(f_out)
        print(f"이미 {len(generated_conversations)}개의 대화가 생성되었습니다.")
    except json.JSONDecodeError:
        print("Output file is corrupted or contains invalid JSON. Starting from scratch.")
        generated_conversations = []
else:
    print("처리된 대화가 없습니다. 처음부터 시작합니다.")

# (topic, emotion) 쌍을 그룹화하여 카운트
conversation_count = defaultdict(int)
for conv in generated_conversations:
    key = (conv['topic'], conv['theme'])
    conversation_count[key] += 1

# 부족한 (topic, emotion) 쌍 식별
incomplete_pairs = []
for topic in topics_data:
    for emotion in emotions:
        key = (topic, emotion)
        if conversation_count.get(key, 0) < num_conversations_per_pair:
            incomplete_pairs.append((key, conversation_count.get(key, 0)))

if not incomplete_pairs:
    print("모든 (topic, emotion) 쌍이 원하는 개수만큼 대화를 생성했습니다.")
else:
    print(f"\n부족한 (topic, emotion) 쌍의 개수: {len(incomplete_pairs)}")
    # tqdm 진행바 설정
    total_missing = sum(num_conversations_per_pair - count for (_, count) in incomplete_pairs)
    pbar = tqdm(total=total_missing, desc="Generating Missing Conversations")

    # Gemini 모델 초기화
    model = genai.GenerativeModel('gemini-1.5-flash-8b')

    for (topic, emotion), current_count in incomplete_pairs:
        needed = num_conversations_per_pair - current_count
        for i in range(needed):
            # 고도화된 프롬프트 생성
            prompt = f"""
As a creative writer, craft a natural and coherent dialogue between two characters about the topic "{topic}". The conversation should vividly convey the emotion of "{emotion}" throughout the interaction. Use the following format strictly without any additional text or explanations:

A: [First character's utterance]
B: [Second character's response]
A: [First character's reply]
...

Ensure that the dialogue captures the essence of "{emotion}" and stays focused on the topic "{topic}". Do not include any narration, descriptions, or emotion labels. Provide only the dialogue in the specified format.
"""

            try:
                # 모델을 사용하여 대화 생성
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        max_output_tokens=1024
                    )
                )

                # 생성된 대화 텍스트 추출 및 전처리
                conversation_text = response.text.strip()

                # 화자 구분과 텍스트만 포함하도록 전처리
                conversation_lines = [
                    line.strip()
                    for line in conversation_text.split('\n')
                    if line.strip().startswith(('A:', 'B:'))
                ]
                conversation_text_cleaned = '\n'.join(conversation_lines)

                # 생성된 대화 저장
                generated_conversations.append({
                    'topic': topic,
                    'theme': emotion,
                    'dialogue': conversation_text_cleaned
                })

                # 대화 수 업데이트
                conversation_count[(topic, emotion)] += 1

                # 진행바 업데이트
                pbar.update(1)

                # 생성된 데이터를 실시간으로 저장 (일정 간격으로 백업 포함)
                if len(generated_conversations) % 100 == 0:
                    # 백업 파일 생성
                    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
                    backup_file = os.path.join(backup_dir, f'synth_gemini_step4_10_backup_{timestamp}.json')
                    shutil.copy(output_file, backup_file)
                    print(f"백업 생성: {backup_file}")

                    # 데이터 저장
                    with open(output_file, 'w', encoding='utf-8') as f_out:
                        json.dump(generated_conversations, f_out, ensure_ascii=False, indent=4)

                # API 호출 제한 준수를 위한 지연 시간 추가
                time.sleep(0.1)

            except Exception as e:
                print(f"Error generating conversation for topic '{topic}' and emotion '{emotion}': {e}")
                # Optional: 재시도 로직 추가 가능
                time.sleep(1)
                continue

    # 진행바 닫기
    pbar.close()

    # 최종 백업 및 데이터 저장
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    backup_file = os.path.join(backup_dir, f'synth_gemini_step4_10_backup_final_{timestamp}.json')
    shutil.copy(output_file, backup_file)
    print(f"최종 백업 생성: {backup_file}")

    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(generated_conversations, f_out, ensure_ascii=False, indent=4)

    print("부족한 대화 생성이 완료되었습니다.")

# 최종 요약 출력
total_conversations = len(generated_conversations)
total_expected_conversations = len(topics_data) * len(emotions) * num_conversations_per_pair
print("\n--- 요약 ---")
print(f"전체 대화 개수: {total_conversations}")
print(f"설계 상 출력되어야 했던 전체 대화 개수: {total_expected_conversations}")
