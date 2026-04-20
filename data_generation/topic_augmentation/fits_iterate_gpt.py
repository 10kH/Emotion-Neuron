import json
import time
import os
from tqdm import tqdm
import openai

# OpenAI API 설정
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

openai.api_key = OPENAI_API_KEY

# 파일 경로 설정
input_file = os.environ.get('INPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/fits_sum.json')
output_file = os.environ.get('OUTPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/fits_step5.json')

# 생성할 대화의 개수 설정 (각 주제 & 감정 쌍당)
num_conversations_per_pair = 1  # 각 주제당 몇 개의 대화를 생성할지 설정

# 주제 데이터 로드
with open(input_file, 'r', encoding='utf-8') as f:
    fits_data = json.load(f)

# 기존 주제 목록 가져오기
existing_topics = set(fits_data.keys())

# 생성된 주제를 저장할 새로운 딕셔너리
modified_fits = {}

# 전체 작업 수 계산
total_tasks = len(fits_data) * num_conversations_per_pair
pbar = tqdm(total=total_tasks, desc="Generating Topics")

# 각 주제에 대해 유사한 주제로 대체
for topic, _ in fits_data.items():
    for i in range(num_conversations_per_pair):
        # 고도화된 프롬프트 생성
        prompt = f"""
Please generate a one- or two-word alternative to the term "{topic}" that is semantically related and could replace it in similar contexts. 

Requirements:
1. The new term should be relevant to "{topic}" and evoke similar associations without being identical in form or concept.
2. Avoid terms that add descriptive phrases or explanations, such as "The History of" or "Overview of." Instead, aim for concise, commonly recognized nouns or phrases.
3. Ensure the alternative is distinct from, and not included in, the following existing list:
{', '.join(existing_topics)}

Example Transformations:
- "New York" → "Los Angeles"
- "Apple Company" → "Microsoft"
- "European Union" → "NATO"
- "Piano" → "Violin"

Provide only the new term without any additional explanations or annotations.
"""

        try:
            # OpenAI GPT API를 사용하여 유사한 주제 생성
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                temperature=1.0,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # 생성된 주제 추출
            new_topic = response['choices'][0]['message']['content'].strip()

            # 중복 확인 및 재시도
            attempts = 1
            max_attempts = 100
            while new_topic in existing_topics and attempts < max_attempts:
                print(f"Duplicate topic '{new_topic}' found. Retrying ({attempts}/{max_attempts})...")
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    temperature=1.0,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                new_topic = response['choices'][0]['message']['content'].strip()
                attempts += 1

            if new_topic in existing_topics:
                print(f"Failed to generate a unique topic for '{topic}' after {max_attempts} attempts. Skipping.")
                modified_fits[topic] = ""
            else:
                # 새로운 주제를 기존 주제 목록에 추가하고, 수정된 딕셔너리에 저장
                existing_topics.add(new_topic)
                modified_fits[new_topic] = ""
                print(f"Replaced '{topic}' with '{new_topic}'.")

        except Exception as e:
            print(f"Error generating a similar topic for '{topic}': {e}")
            modified_fits[topic] = ""  # 오류 발생 시 원본 주제 유지

        # 진행 상황 업데이트 및 실시간 저장
        pbar.update(1)
        with open(output_file, 'w', encoding='utf-8') as f_out:
            json.dump(modified_fits, f_out, ensure_ascii=False, indent=4)

# tqdm 진행바 닫기
pbar.close()

print("Topic replacement completed successfully.")
