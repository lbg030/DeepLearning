import json 
from pathlib import Path 
import argparse
import os

def parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, default = "/Users/gwonsmpro/Desktop/lbg",help="십시일반 폴더의 경로를 넣어주세요.")
    args = parser.parse_args()
    return args

def main(args):
    label_paths = list(Path(args.dir).glob("*.json"))
    n_labels = len(label_paths)
    dolgi_count, slit_count, both_count = 0, 0, 0
    for p in label_paths:
        with open(p, 'r') as f:
            label_dict = json.load(f)
            f.close()
        
        labels = []
        if not label_dict["shapes"]:
            os.remove(p)
            print(p)
        for label in label_dict["shapes"]:
            labels.append(label["label"])
        if "dolgi" in labels:
            dolgi_count += 1 
        elif "slit" in labels:
            slit_count += 1
        elif ("dolgi" in labels) & ("slit" in labels):
            both_count += 1
    
    print(f"라벨링 데이터: {n_labels}개")
    print(f"돌기 데이터 갯수: {dolgi_count}개")
    print(f"슬릿 데이터 갯수: {slit_count}개")
    print(f"둘다 있는 데이터 갯수: {both_count}개")

if __name__=="__main__":
    args = parser()
    # parser를 사용하지 않을시, 아래 예시와 같이 args.dir에 십시일반 데이터가 위치한 경로를 넣어주세요 
    # 예시
    # args.dir = "/Users/injo/Library/Mobile Documents/com~apple~CloudDocs/injo/업무/RTM/projects/dd_aoi/data/십시일반_라벨링_230208/kij"
    main(args)