import json
import time
import os
from tqdm import tqdm
import google.generativeai as genai

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

# 주제 데이터 로드
with open(input_file, 'r', encoding='utf-8') as f:
    topics_data = json.load(f)

# 생성된 대화 저장을 위한 리스트 초기화
generated_conversations = []

# 기존에 생성된 데이터가 있다면 로드하여 이어서 작업
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

# 기존 대화를 (topic, emotion) 기준으로 그룹화하여 카운트
conversation_count = defaultdict(int)
for conv in generated_conversations:
    key = (conv['topic'], conv['theme'])
    conversation_count[key] += 1

# 전체 작업 수 및 완료된 작업 수 계산
total_tasks = len(topics_data) * len(emotions) * num_conversations_per_pair
completed_tasks = sum(conversation_count.values())
remaining_tasks = total_tasks - completed_tasks

# tqdm 진행바 설정 (initial을 사용하여 이미 완료된 작업 수 설정)
pbar = tqdm(total=total_tasks, initial=completed_tasks, desc="Generating Conversations")

# Gemini 모델 초기화
model = genai.GenerativeModel('gemini-1.5-flash-8b')

# 각 주제와 감정 쌍에 대해 대화 생성
for topic in topics_data:
    for emotion in emotions:
        key = (topic, emotion)
        generated = conversation_count.get(key, 0)
        if generated >= num_conversations_per_pair:
            continue  # 이미 충분한 대화가 생성된 경우 건너뜀
        for i in range(generated, num_conversations_per_pair):
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

                # 생성된 데이터를 실시간으로 저장
                with open(output_file, 'w', encoding='utf-8') as f_out:
                    json.dump(generated_conversations, f_out, ensure_ascii=False, indent=4)

                # API 호출 제한 준수를 위한 지연 시간 추가
                time.sleep(0.1)

                # 대화 수 업데이트
                conversation_count[key] += 1

            except Exception as e:
                print(f"Error generating conversation for topic '{topic}' and emotion '{emotion}' (Attempt {i+1}/{num_conversations_per_pair}): {e}")
                continue

            # 진행바 업데이트
            pbar.update(1)

# tqdm 진행바 닫기
pbar.close()

print("Data generation completed successfully.")