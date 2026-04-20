import json
import os
from collections import defaultdict
from tqdm import tqdm

def load_json_file(file_path):
    """
    지정된 경로의 JSON 파일을 로드하여 데이터를 반환합니다.
    
    Args:
        file_path (str): JSON 파일의 경로.
    
    Returns:
        list: JSON 파일의 데이터 리스트.
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def find_duplicate_dialogues(data):
    """
    대화 데이터에서 중복되는 'dialogue' 항목을 찾습니다.
    
    Args:
        data (list): 대화 데이터 리스트.
    
    Returns:
        dict: 중복된 'dialogue'와 그 등장 횟수를 포함하는 딕셔너리.
        dict: 중복된 'dialogue'와 해당 대화의 인덱스 리스트를 포함하는 딕셔너리.
    """
    dialogue_counter = defaultdict(int)
    dialogue_indices = defaultdict(list)
    
    for idx, entry in enumerate(tqdm(data, desc="중복 검사 중")):
        dialogue = entry.get('dialogue', '').strip()
        if dialogue:
            dialogue_counter[dialogue] += 1
            dialogue_indices[dialogue].append(idx)
    
    # 중복된 대화만 필터링
    duplicates = {dialogue: count for dialogue, count in dialogue_counter.items() if count > 1}
    duplicates_indices = {dialogue: indices for dialogue, indices in dialogue_indices.items() if len(indices) > 1}
    
    return duplicates, duplicates_indices

def save_duplicates(duplicates, duplicates_indices, output_path):
    """
    중복된 대화 정보를 JSON 파일로 저장합니다.
    
    Args:
        duplicates (dict): 중복된 대화와 그 등장 횟수를 포함하는 딕셔너리.
        duplicates_indices (dict): 중복된 대화와 해당 대화의 인덱스 리스트를 포함하는 딕셔너리.
        output_path (str): 저장할 JSON 파일의 경로.
    """
    duplicates_data = []
    for dialogue, count in duplicates.items():
        duplicates_data.append({
            "dialogue": dialogue,
            "count": count,
            "indices": duplicates_indices[dialogue]
        })
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(duplicates_data, f, ensure_ascii=False, indent=4)
    print(f"중복된 대화 정보를 '{output_path}'에 저장했습니다.")

def main():
    # 파일 경로 설정
    input_file = os.environ.get('INPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/emoprism.json')
    output_file = os.environ.get('OUTPUT_FILE', '/home/woody/workspace/Emotion-Neuron/data/IO_duplicates.json')
    
    # 입력 파일 존재 여부 확인
    if not os.path.exists(input_file):
        print(f"입력 파일을 찾을 수 없습니다: {input_file}")
        return
    
    # JSON 데이터 로드
    print("JSON 파일을 로드하는 중...")
    data = load_json_file(input_file)
    print(f"총 {len(data)}개의 대화를 로드했습니다.")
    
    # 중복 대화 찾기
    print("중복 대화를 찾는 중...")
    duplicates, duplicates_indices = find_duplicate_dialogues(data)
    
    if not duplicates:
        print("중복된 대화가 존재하지 않습니다.")
    else:
        print(f"총 {len(duplicates)}개의 중복된 대화가 발견되었습니다.")
        # 중복 대화 상세 정보 출력
        for dialogue, count in duplicates.items():
            print(f"\n감정: {data[duplicates_indices[dialogue][0]].get('theme', 'N/A')}")
            print(f"대화 내용:\n{dialogue}")
            print(f"등장 횟수: {count}")
            print(f"인덱스: {duplicates_indices[dialogue]}")
        
        # 중복 대화 정보를 파일로 저장할지 여부 결정
        save_to_file = input("\n중복된 대화 정보를 JSON 파일로 저장하시겠습니까? (y/n): ").strip().lower()
        if save_to_file == 'y':
            save_duplicates(duplicates, duplicates_indices, output_file)

if __name__ == "__main__":
    main()
