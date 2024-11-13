import cv2
import json
import time
from ultralytics import YOLO
import threading

def setup_model():
    try:
        # YOLOv8のモデルをロード
        model = YOLO('./yolov8n.pt')  # 軽量モデル yolov8n
        return model
    except Exception as e:
        print(f"モデルの読み込みに失敗しました: {e}")
        return None

def load_regions_from_json(json_file):
    try:
        with open(json_file, 'r') as file:
            data = json.load(file)
            return data['regions']
    except Exception as e:
        print(f"JSONファイルの読み込みに失敗しました: {e}")
        return []

def is_person_in_region(person_box, region_coords):
    x1, y1, x2, y2 = region_coords
    # 人物のバウンディングボックス
    px1, py1, px2, py2 = person_box
    # バウンディングボックスが領域内に重なっているか判定
    return not (px2 < x1 or px1 > x2 or py2 < y1 or py1 > y2)

def detect_person_in_regions(model, frame, regions, person_stay_info, threshold_seconds):
    # 推論の実行（人物クラスの追跡）
    results = model.track(source=frame, persist=True)

    # 人物クラスID 0 でフィルタリング
    person_results = [result for result in results[0].boxes if result.cls == 0]

    current_time = time.time()

    for region in regions:
        region_coords = region['coordinates']
        region_id = region['id']
        region_occupied = False  # 領域内に人物がいるかどうかのフラグ

        # 指定された領域を赤色で囲む
        x1, y1, x2, y2 = region_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for person in person_results:
            person_id = person.id  # YOLOのトラッキングID
            
            if person_id == None:
                print("person_id is None")
            else:
                person_id = int(person_id)
            person_box = person.xyxy[0].cpu().numpy().astype(int)

            x_min, y_min, x_max, y_max = person_box  # 左上と右下の座標

            color = (0, 255, 0)  # 緑色の矩形
            thickness = 2  # 矩形の線の太さ

            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), color, thickness)
            
            if is_person_in_region(person_box, region_coords):
                # 人物が領域内にいる場合
                region_occupied = True
                if person_id not in person_stay_info:
                    person_stay_info[person_id] = {'region_id': region_id, 'start_time': current_time}
                else:
                    # 同じ領域にいるか確認
                    if person_stay_info[person_id]['region_id'] == region_id:
                        stay_duration = current_time - person_stay_info[person_id]['start_time']
                        if stay_duration >= threshold_seconds:
                            print(f"{region_id}:の領域に人物が{threshold_seconds}秒以上滞在しています")
                    else:
                        # 別の領域に移動した場合、滞在時間をリセット
                        person_stay_info[person_id] = {'region_id': region_id, 'start_time': current_time}
            else:
                # 領域外に出た場合、カウントをリセット
                if person_id in person_stay_info and person_stay_info[person_id]['region_id'] == region_id:
                    try:
                        if person_stay_info[person_id]['exit_time']:
                            pass
                    except:
                        person_stay_info[person_id]['exit_time'] = current_time

                    if current_time - person_stay_info[person_id]['exit_time'] > 3:
                        del person_stay_info[person_id]
                        print("離席判定")
                    pass

        if not region_occupied:
            # 領域に人物がいない場合、カウントをリセットし、メッセージを表示
            print(f"{region_id}:の領域に人物は存在しません")
            # 特定の領域に対応する人物のカウントをリセット
            for person_id, stay_info in list(person_stay_info.items()):
                if stay_info['region_id'] == region_id:
                    try:
                        if person_stay_info[person_id]['exit_time']:
                            pass
                    except:
                        person_stay_info[person_id]['exit_time'] = current_time
                        pass
                    
                    if current_time - person_stay_info[person_id]['exit_time'] > 3:
                        del person_stay_info[person_id]
        #                 print("離席判定")


def detect_person_from_camera(cam_id, json_file, threshold_seconds=5):
    # モデルのセットアップ
    model = setup_model()
    if model is None:
        print("モデルがロードされていないため、処理を終了します。")
        return

    # JSONファイルから領域データを読み込み
    regions = load_regions_from_json(json_file)
    if not regions:
        print("領域データがないため、処理を終了します。")
        return

    # カメラからの映像取得を開始
    cap = cv2.VideoCapture(cam_id)

    if not cap.isOpened():
        print("カメラの起動に失敗しました。")
        return

    # 各人物の滞在時間を記録する辞書
    person_stay_info = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("フレームの取得に失敗しました。")
                break

            # 指定された領域に人物がいるかどうか判定
            detect_person_in_regions(model, frame, regions, person_stay_info, threshold_seconds)

            # 映像をリアルタイムで表示
            cv2.imshow(f'Person Detection (Press "q" to quit), cam_id={str(cam_id)}', frame)

            # 'q'キーが押されたらループを終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except Exception as e:
        print(f"エラーが発生しました: {e}")
    finally:
        # カメラの解放とウィンドウの破棄
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # JSONファイルのパスを指定
    json_file = 'regions.json'
    cam_id = 0
    t1 = threading.Thread(target=detect_person_from_camera, args=(cam_id, json_file), name='Thread T1', daemon=True)
    cam_id = 1
    t2 = threading.Thread(target=detect_person_from_camera, args=(cam_id, json_file), name='Thread T2', daemon=True)
    
    t1.start()
    t2.start()

    t1.join()
    t2.join()