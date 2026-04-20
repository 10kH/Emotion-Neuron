import json
import os
import time
import shutil
from tqdm import tqdm
import anthropic

# Anthropic API 키 설정
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY 환경 변수를 설정해주세요.")

# Anthropic 클라이언트 초기화
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# 감정 리스트 정의
emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# 입력 및 출력 파일 경로 설정
input_file = os.environ.get('INPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/synth_gemini_step4_10.json')
output_file = os.environ.get('OUTPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/synth_gemini_step4_10_labeled_claude.json')

# 입력 파일에서 대화 데이터 로드
with open(input_file, 'r', encoding='utf-8') as f:
    dialogues = json.load(f)

# 출력 파일이 존재하면 기존 레이블링된 데이터 로드
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f_out:
        labeled_dialogues = json.load(f_out)
    # 이미 레이블링된 대화의 인덱스 저장
    labeled_indices = {i for i, d in enumerate(labeled_dialogues) if "claude-3-haiku-20240307" in d}
    print(f"{len(labeled_indices)}개의 대화가 이미 레이블링되었습니다.")
else:
    labeled_dialogues = []
    labeled_indices = set()
    print("처리된 대화가 없습니다. 처음부터 시작합니다.")

# 전체 대화 수
total_dialogues = len(dialogues)

# 전체 작업 수에 대한 진행 바 설정
pbar = tqdm(total=total_dialogues, desc="감정 레이블링 진행 중", unit="대화")

# 이미 레이블링된 항목에 대해 진행 바 업데이트
pbar.update(len(labeled_indices))

# 모든 대화를 순회하며 레이블링 수행
for idx, dialogue in enumerate(dialogues):
    # 이미 레이블링된 경우 건너뜀
    if idx in labeled_indices:
        continue

    # 프롬프트 작성
    system_prompt = "You are a sentiment analysis expert."
    user_prompt = f"""
Analyze the following conversation and determine the primary emotion that best represents the overall sentiment expressed throughout the dialogue.

Select only from the following emotions: {', '.join(emotions)}.

Conversation:
{dialogue['dialogue']}

Please respond with only the emotion name from the provided list. Do not include any additional text, formatting, or explanations.
"""

    # 메시지 구성
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_prompt
                }
            ]
        }
    ]

    try:
        # Claude 모델을 사용하여 감정 예측
        response = client.messages.create(
            model="claude-3-haiku-20240307",
            max_tokens=10,
            temperature=0.0,
            system=system_prompt,
            messages=messages
        )

        # 예측된 감정 추출 및 전처리
        predicted_emotion = ''.join([block.text for block in response.content]).strip().lower()

        # 유효한 감정인지 확인
        if predicted_emotion in emotions:
            dialogue["claude-3-haiku-20240307"] = predicted_emotion
        else:
            print(f"유효하지 않은 감정 예측: '{predicted_emotion}' (주제: '{dialogue.get('topic', 'unknown')}')")
            dialogue["claude-3-haiku-20240307"] = "unknown"

    except Exception as e:
        # 에러 발생 시 'error'로 레이블링
        print(f"감정 레이블링 중 에러 발생 (주제: '{dialogue.get('topic', 'unknown')}'): {e}")
        dialogue["claude-3-haiku-20240307"] = "error"

    # 레이블링된 대화를 추가
    labeled_dialogues.append(dialogue)

    # 실시간으로 출력 파일에 저장
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(labeled_dialogues, f_out, ensure_ascii=False, indent=4)

    # 진행 바 업데이트
    pbar.update(1)

    # API 호출 제한을 피하기 위해 짧은 지연 시간 추가
    time.sleep(0.1)

# 진행 바 닫기
pbar.close()

print("감정 레이블링이 성공적으로 완료되었습니다.")
