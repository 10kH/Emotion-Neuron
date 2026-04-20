import json
from collections import defaultdict

def count_conversations(json_file_path):
    """
    지정된 JSON 파일의 대화 항목 수를 세고, 각 (topic, emotion) 쌍별로 개수를 출력합니다.
    
    :param json_file_path: 대화가 저장된 JSON 파일의 경로
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            conversations = json.load(f)
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {json_file_path}")
        return
    except json.JSONDecodeError:
        print(f"파일이 올바른 JSON 형식이 아닙니다: {json_file_path}")
        return
    
    total_conversations = len(conversations)
    print(f"전체 대화 개수: {total_conversations}")
    
    # (topic, emotion) 쌍별로 개수 집계
    count_dict = defaultdict(int)
    for conv in conversations:
        key = (conv.get('topic'), conv.get('theme'))
        count_dict[key] += 1
    
    print("\n(topic, emotion) 쌍별 대화 개수:")
    for key, count in count_dict.items():
        topic, emotion = key
        print(f"주제: '{topic}', 감정: '{emotion}' - 개수: {count}")
    
    # 원하는 개수만큼 생성되었는지 확인
    num_conversations_per_pair = 10  # 원하는 값으로 변경하세요
    incomplete_pairs = []
    total_expected_conversations = 0

    for key, count in count_dict.items():
        total_expected_conversations += num_conversations_per_pair
        if count < num_conversations_per_pair:
            incomplete_pairs.append((key, count))
    
    if not incomplete_pairs:
        print("\n모든 (topic, emotion) 쌍이 원하는 개수만큼 대화를 생성했습니다.")
    else:
        print("\n아래의 (topic, emotion) 쌍은 원하는 개수만큼 대화가 생성되지 않았습니다:")
        for key, count in incomplete_pairs:
            topic, emotion = key
            print(f"주제: '{topic}', 감정: '{emotion}' - 현재 개수: {count}, 필요한 개수: {num_conversations_per_pair}")
    
    # 전체 대화 개수와 설계 상 출력되어야 했던 개수 출력
    print("\n--- 요약 ---")
    print(f"전체 대화 개수: {total_conversations}")
    print(f"설계 상 출력되어야 했던 전체 대화 개수: {total_expected_conversations}")

if __name__ == "__main__":
    import os
    # 확인하고자 하는 JSON 파일의 경로를 지정하세요
    json_file = os.environ.get('INPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/synth_gemini_step4_10.json')
    count_conversations(json_file)