import json
import os
import time
import shutil
from tqdm import tqdm
import anthropic
from datetime import datetime

# Anthropic API 키 설정
ANTHROPIC_API_KEY = os.environ.get('ANTHROPIC_API_KEY')
if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY 환경 변수를 설정해주세요.")

# Anthropic 클라이언트 초기화
client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

# 감정 리스트 정의
emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]

# 입력 및 출력 파일 경로 설정
output_file = os.environ.get('OUTPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/synth_gemini_step4_10_labeled_claude.json')

# 백업 디렉토리 설정
backup_dir = os.environ.get('BACKUP_DIR', '/home/woody/workspace/Emotion-Neuron/data/backup_claude')
os.makedirs(backup_dir, exist_ok=True)

# 현재 시간으로 타임스탬프 생성
timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
backup_file = os.path.join(backup_dir, f'synth_gemini_step4_10_labeled_claude_backup_{timestamp}.json')

# 기존 출력 파일이 존재하는지 확인 및 백업 생성
if os.path.exists(output_file):
    try:
        shutil.copy(output_file, backup_file)
        print(f"기존 출력 파일을 백업했습니다: {backup_file}")
    except Exception as e:
        raise IOError(f"백업 생성 중 오류 발생: {e}")

    # 출력 파일에서 레이블링된 데이터 로드
    try:
        with open(output_file, 'r', encoding='utf-8') as f_out:
            labeled_dialogues = json.load(f_out)
        print(f"총 {len(labeled_dialogues)}개의 대화가 로드되었습니다.")
    except json.JSONDecodeError:
        raise ValueError(f"출력 파일이 올바른 JSON 형식이 아닙니다: {output_file}")
else:
    raise FileNotFoundError(f"출력 파일을 찾을 수 없습니다: {output_file}")

# 재레이블링 대상 정의
relabeling_targets = ["unknown", "error"]

for target_label in relabeling_targets:
    # 현재 타겟 레이블을 가진 대화만 선택
    dialogues_to_relabel = [
        dialogue for dialogue in labeled_dialogues
        if dialogue.get("claude-3-haiku-20240307") == target_label
    ]
    total_to_relabel = len(dialogues_to_relabel)

    if total_to_relabel == 0:
        print(f"재레이블링할 '{target_label}' 감정을 가진 대화가 없습니다.")
        continue

    print(f"\n재레이블링할 '{target_label}' 감정을 가진 대화 수: {total_to_relabel}")

    # tqdm 진행바 설정
    pbar = tqdm(total=total_to_relabel, desc=f"감정 재레이블링 ({target_label}) 진행 중", unit="대화")

    # 주기적인 백업 설정
    BACKUP_INTERVAL = 100  # 매 100개마다 백업 생성
    processed_count = 0

    # 모든 대화를 순회하며 'unknown' 또는 'error'인 항목만 재레이블링
    for dialogue in dialogues_to_relabel:
        # 프롬프트 작성
        system_prompt = "You are a sentiment analysis expert."
        user_prompt = f"""
Analyze the following conversation and determine the primary emotion that best represents the overall sentiment expressed throughout the dialogue.

**Important: You must select only one emotion from the following list and respond with exactly one word in lowercase. Do not use any other emotion words or additional text, formatting, explanations, or comments.**

List of emotions:
{', '.join(emotions)}.

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
                temperature=0.3,
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
            print(f"감정 재레이블링 중 에러 발생 (주제: '{dialogue.get('topic', 'unknown')}'): {e}")
            dialogue["claude-3-haiku-20240307"] = "error"

        # 실시간으로 출력 파일에 저장
        try:
            with open(output_file, 'w', encoding='utf-8') as f_out:
                json.dump(labeled_dialogues, f_out, ensure_ascii=False, indent=4)
        except Exception as e:
            print(f"데이터 저장 중 오류 발생: {e}")

        # 진행 바 업데이트
        pbar.update(1)
        processed_count += 1

        # 주기적으로 백업 생성
        if processed_count % BACKUP_INTERVAL == 0:
            current_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            current_backup_file = os.path.join(backup_dir, f'synth_gemini_step4_10_labeled_claude_backup_{current_timestamp}.json')
            try:
                shutil.copy(output_file, current_backup_file)
                print(f"주기적인 백업 생성: {current_backup_file}")
            except Exception as e:
                print(f"주기적인 백업 생성 중 오류 발생: {e}")

        # API 호출 제한을 피하기 위해 짧은 지연 시간 추가
        time.sleep(0.1)

    # 진행 바 닫기
    pbar.close()

    print(f"'{target_label}' 레이블의 재레이블링이 완료되었습니다.")

# 최종 백업 생성
final_timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
final_backup_file = os.path.join(backup_dir, f'synth_gemini_step4_10_labeled_claude_backup_final_{final_timestamp}.json')
try:
    shutil.copy(output_file, final_backup_file)
    print(f"최종 백업 생성: {final_backup_file}")
except Exception as e:
    print(f"최종 백업 생성 중 오류 발생: {e}")

print("재레이블링이 성공적으로 완료되었습니다.")
