import json
import os
from collections import Counter, defaultdict
from tqdm import tqdm

def load_json_file(file_path):
    """JSON 파일을 로드하여 리스트를 반환합니다."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def merge_files(file_paths):
    """여러 JSON 파일을 하나의 리스트로 병합합니다."""
    merged_data = []
    for file_path in file_paths:
        data = load_json_file(file_path)
        merged_data.extend(data)
    return merged_data

def compute_statistics(dialogues):
    """
    대화 리스트를 받아 감정별, 레이블별 통계를 계산합니다.
    반환값은 감정 카운트, 레이블 카운트, 총 대화 수입니다.
    """
    emotion_counts = Counter()
    label_counts = Counter()
    
    for dialogue in dialogues:
        # 각 모델 및 레이블의 감정 카운트
        for key in dialogue:
            if key in ["theme", "label"] or "emotion" in key or "claude" in key or "gpt" in key or "gemini" in key:
                emotion = dialogue.get(key, "").lower()
                if emotion in emotions:
                    emotion_counts[emotion] += 1
                elif emotion == "unknown" or emotion == "error":
                    emotion_counts[emotion] += 1
        
        # 'label' 필드 카운트
        label = dialogue.get("label", "").lower()
        if label in emotions:
            label_counts[label] += 1
        elif label == "unknown" or label == "error":
            label_counts[label] += 1
    
    total_dialogues = len(dialogues)
    return emotion_counts, label_counts, total_dialogues

def print_statistics(stats, title):
    """통계 정보를 표 형식으로 출력합니다."""
    emotion_counts, label_counts, total = stats
    print(f"\n=== {title} ===")
    print(f"총 대화 수: {total}\n")
    
    print("감정별 분포:")
    print(f"{'Emotion':<15}{'Count':<10}{'Percentage':<10}")
    for emotion, count in emotion_counts.items():
        percentage = (count / total) * 100
        print(f"{emotion:<15}{count:<10}{percentage:.2f}%")
    
    print("\n레이블별 분포:")
    print(f"{'Label':<15}{'Count':<10}{'Percentage':<10}")
    for label, count in label_counts.items():
        percentage = (count / total) * 100
        print(f"{label:<15}{count:<10}{percentage:.2f}%")

def main():
    # 감정 리스트 정의
    global emotions
    emotions = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
    
    # 입력 파일 경로 목록
    _synth_dir = os.environ.get('SYNTH_DIR', '/home/woody/workspace/IO/Data/Synth')
    input_files = [
        os.path.join(_synth_dir, 'step0.json'),
        os.path.join(_synth_dir, 'step1.json'),
        os.path.join(_synth_dir, 'step2.json'),
        os.path.join(_synth_dir, 'step3.json'),
        os.path.join(_synth_dir, 'step4.json'),
    ]

    # 출력 파일 경로
    output_file = os.environ.get('OUTPUT_FILE', os.path.join(_synth_dir, 'emoprism.json'))

    # 통계 파일 경로
    statistics_file = os.environ.get('STATS_FILE', os.path.join(_synth_dir, 'emoprism_stats.json'))
    
    # 파일 병합
    print("파일을 병합하는 중...")
    merged_dialogues = merge_files(input_files)
    print(f"총 {len(merged_dialogues)}개의 대화가 병합되었습니다.")
    
    # 병합된 데이터를 저장 (선택 사항)
    with open(output_file, 'w', encoding='utf-8') as f_out:
        json.dump(merged_dialogues, f_out, ensure_ascii=False, indent=4)
    print(f"병합된 데이터를 '{output_file}'에 저장했습니다.")
    
    # 개별 파일 통계 계산
    individual_stats = {}
    for file_path in input_files:
        file_name = os.path.basename(file_path)
        print(f"\n파일 '{file_name}'의 통계를 계산하는 중...")
        dialogues = load_json_file(file_path)
        stats = compute_statistics(dialogues)
        individual_stats[file_name] = {
            "emotion_counts": dict(stats[0]),
            "label_counts": dict(stats[1]),
            "total_dialogues": stats[2]
        }
        print_statistics(stats, f"'{file_name}'의 통계")
    
    # 전체 데이터 통계 계산
    print("\n전체 병합된 데이터의 통계를 계산하는 중...")
    overall_stats = compute_statistics(merged_dialogues)
    print_statistics(overall_stats, "전체 데이터의 통계")
    
    # 통계 결과를 JSON 파일로 저장 (선택 사항)
    stats_output = {
        "individual_file_stats": individual_stats,
        "overall_stats": {
            "emotion_counts": dict(overall_stats[0]),
            "label_counts": dict(overall_stats[1]),
            "total_dialogues": overall_stats[2]
        }
    }
    
    with open(statistics_file, 'w', encoding='utf-8') as f_stat:
        json.dump(stats_output, f_stat, ensure_ascii=False, indent=4)
    print(f"\n통계 결과를 '{statistics_file}'에 저장했습니다.")

if __name__ == "__main__":
    main()
