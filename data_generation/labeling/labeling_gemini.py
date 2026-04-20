import json
import os
import time
from tqdm import tqdm
import google.generativeai as genai

# Gemini API 설정
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경 변수를 설정해주세요.")

# Gemini API 키 설정
genai.configure(api_key=GOOGLE_API_KEY)

# 감정 리스트 정의
emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# 입력 및 출력 파일 경로 설정
input_file = os.environ.get('INPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/synth_gemini_step4_10.json')
output_file = os.environ.get('OUTPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/synth_gemini_step4_10_labeled_gemini.json')

# 입력 파일에서 대화 데이터 로드
with open(input_file, 'r', encoding='utf-8') as f:
    dialogues = json.load(f)

# 출력 파일이 존재하면 기존 레이블링된 데이터 로드
if os.path.exists(output_file):
    with open(output_file, 'r', encoding='utf-8') as f_out:
        labeled_dialogues = json.load(f_out)
    # 이미 레이블링된 대화의 인덱스 저장
    labeled_indices = {i for i, d in enumerate(labeled_dialogues) if "gemini-1.5-flash" in d}
    print(f"{len(labeled_indices)}개의 대화가 이미 레이블링되었습니다.")
else:
    labeled_dialogues = []
    labeled_indices = set()
    print("처리된 대화가 없습니다. 처음부터 시작합니다.")

# 전체 대화 수
total_dialogues = len(dialogues)

# 전체 작업 수에 대한 진행바 설정
pbar = tqdm(total=total_dialogues, desc="감정 레이블링 진행 중", unit="대화")

# 이미 레이블링된 항목에 대해 진행바 업데이트
pbar.update(len(labeled_indices))

# Gemini 모델 초기화
model = genai.GenerativeModel('gemini-1.5-flash')

# 모든 대화를 순회하며 레이블링 수행
for idx, dialogue in enumerate(dialogues):
    # 이미 레이블링된 경우 건너뜀
    if idx in labeled_indices:
        continue

    # 프롬프트 작성 (전문적이고 정제된 형태)
    prompt = f"""
You are a sentiment analysis expert. Analyze the following conversation and determine the primary emotion that best represents the overall sentiment expressed throughout the dialogue. Select only from the following emotions: {', '.join(emotions)}.

Conversation:
{dialogue['dialogue']}

Please respond with only the emotion name from the provided list. Do not include any additional text, formatting, or explanations.
"""

    try:
        # Gemini 모델을 사용하여 감정 예측
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,  # 감정 예측의 일관성을 높이기 위해 낮은 온도 사용
                max_output_tokens=10
              ),  # 짧은 응답을 유도
            safety_settings={
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE, 
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE, 
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE, 
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE
                }
            )

        # 예측된 감정 추출 및 전처리
        predicted_emotion = response.text.strip().lower()

        # 유효한 감정인지 확인
        if predicted_emotion in emotions:
            dialogue["gemini-1.5-flash"] = predicted_emotion
        else:
            print(f"유효하지 않은 감정 예측: '{predicted_emotion}' (주제: '{dialogue['topic']}')")
            dialogue["gemini-1.5-flash"] = "unknown"

    except Exception as e:
        # 에러 발생 시 'error'로 레이블링
        print(f"감정 레이블링 중 에러 발생 (주제: '{dialogue['topic']}'): {e}")
        dialogue["gemini-1.5-flash"] = "error"

    # 레이블링된 대화를 추가
    labeled_dialogues.append(dialogue)

    # 실시간으로 출력 파일에 저장
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(labeled_dialogues, f_out, ensure_ascii=False, indent=4)

    # 진행바 업데이트
    pbar.update(1)

    # API 호출 제한을 피하기 위해 짧은 지연 시간 추가
    time.sleep(0.1)

# 진행바 닫기
pbar.close()

print("감정 레이블링이 성공적으로 완료되었습니다.")