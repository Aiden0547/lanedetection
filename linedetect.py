import cv2
import numpy as np
from datetime import datetime
import os

# =========================================================================
# ⚙️ [실험 설정 및 전역 변수] - 지정하신 경로 고정 적용
# =========================================================================
VIDEO_PATH = r"영상경로입력"
LOG_FILE_PATH = r"로그경로입력"

paused = True 
clicked_point = None

weather_mode = 'normal'   # 'normal', 'fog', 'rain'
filter_mode = 'none'      # 'none', 'total', 'fog_spec', 'rain_spec'

total_frames = 0
fail_frames_count = 0


# -------------------------
# 마우스 이벤트 (좌표 디버깅용)
# -------------------------
def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        print(f"클릭 좌표: X={x}, Y={y}")


# -------------------------
# 환경 시뮬레이터 (Noise Generator)
# -------------------------
def apply_weather_noise(frame, mode):
    if mode == 'normal':
        return frame.copy()
        
    elif mode == 'fog':
        fog_layer = np.full(frame.shape, 200, dtype=np.uint8)
        foggy_frame = cv2.addWeighted(frame, 0.5, fog_layer, 0.5, 0)
        return foggy_frame
        
    elif mode == 'rain':
        noisy_frame = frame.copy()
        h, w, c = noisy_frame.shape
        num_noise_pixels = int(h * w * 0.06)
        for _ in range(num_noise_pixels):
            y = np.random.randint(0, h)
            x = np.random.randint(0, w)
            noisy_frame[y, x] = [255, 255, 255]
            
        noisy_frame = cv2.GaussianBlur(noisy_frame, (3, 3), 0)
        return noisy_frame
        
    return frame


# -------------------------
# 3가지 가상 해결 필터 (Advanced Filters)
# -------------------------
def apply_lane_filter(frame, mode):
    if mode == 'none':
        return frame.copy()

    elif mode == 'total':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced_gray = clahe.apply(gray)
        blurred = cv2.GaussianBlur(enhanced_gray, (7, 7), 0)
        return cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)

    elif mode == 'fog_spec':
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)

    elif mode == 'rain_spec':
        denoised = cv2.medianBlur(frame, 5)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)
        return dilated

    return frame


# -------------------------
# ROI 설정
# -------------------------
def region_of_interest(img):
    polygon = np.array([[[2, 718], [553, 524], [761, 524], [1118, 719]]], np.int32)
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, polygon, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked


# -------------------------
# 차선 검출 및 오류 추론
# -------------------------
def detect_lane(frame):
    global fail_frames_count

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    roi = region_of_interest(edges)

    white_pixels = cv2.countNonZero(roi)
    
    lines = cv2.HoughLinesP(
        roi,
        rho=2,
        theta=np.pi / 180,
        threshold=80,       
        minLineLength=90,   
        maxLineGap=40       
    )

    line_image = np.zeros_like(frame)
    is_failed = False

    if white_pixels < 300 or lines is None or len(lines) < 2:
        if not paused:
            is_failed = True
            fail_frames_count += 1
    else:
        valid_lines_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) > 0:
                slope = (y2 - y1) / (x2 - x1)
                if abs(slope) < 0.45 or abs(slope) > 2.5:
                    continue
            
            cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
            valid_lines_count += 1
            
        if valid_lines_count < 2 and not paused:
            is_failed = True
            fail_frames_count += 1

    result = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    return result, edges, roi, is_failed


# -------------------------
# UI 출력 함수들
# -------------------------
def draw_click_coordinate(frame):
    if clicked_point is None: return frame
    x, y = clicked_point
    cv2.circle(frame, (x, y), 6, (0, 0, 255), -1)
    cv2.putText(frame, f"({x},{y})", (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return frame

def draw_roi(frame):
    roi_polygon = np.array([[(2, 718), (553, 524), (761, 524), (1118, 719)]], np.int32)
    cv2.polylines(frame, roi_polygon, True, (255, 255, 0), 2)
    return frame


# -------------------------
# 메인 루프 실행
# -------------------------
def main():
    global paused, weather_mode, filter_mode, total_frames, fail_frames_count

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        print(f"동영상 열기 실패: {VIDEO_PATH}")
        return

    cv2.namedWindow("Lane Detection")
    cv2.setMouseCallback("Lane Detection", mouse_callback)

    current_frame = None

    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (1280, 720))
        current_frame = frame.copy()

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("-> 영상 주행 완료")
                break
            frame = cv2.resize(frame, (1280, 720))
            current_frame = frame.copy()
            total_frames += 1

        if current_frame is None: continue

        noisy_frame = apply_weather_noise(current_frame, weather_mode)
        filtered_frame = apply_lane_filter(noisy_frame, filter_mode)
        result, edges, roi, is_failed = detect_lane(filtered_frame)

        result = draw_roi(result)
        result = draw_click_coordinate(result)

        # -----------------------------------------------------------------
        # 📺 [HUD 안내 문구 및 스크립트 대폭 복구]
        # -----------------------------------------------------------------
        # 1. 프로그램 구동 핵심 상태 (READY / ANALYZING)
        state_color = (0, 165, 255) if paused else (0, 255, 0)
        state_text = f"STATE: {'READY (PAUSED)' if paused else 'ANALYZING (PLAYING)'}"
        cv2.putText(result, state_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        
        # 2. 현재 활성화된 세팅 정보 요약
        cv2.putText(result, f"WEATHER : {weather_mode.upper()}", (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"FILTER  : {filter_mode.upper()}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # 3. 전체 조작 단축키 매뉴얼 리스트업 (가독성을 위한 간략 버전 콤팩트 배치)
        cv2.putText(result, "---------------------------------------------", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        cv2.putText(result, "SPACE : Pause / Resume | ESC : Exit & Save", (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, "[Weather] 1: Normal  | 2: Fog  | 3: Rain", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(result, "[Filter]  4: Off | 5: Total | 6: Fog-Spec | 7: Rain-Spec", (20, 185), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(result, "---------------------------------------------", (20, 205), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

        # 4. 정량적 분석 실시간 지표 데이터
        error_rate = (fail_frames_count / total_frames) * 100 if total_frames > 0 else 0
        cv2.putText(result, f"Total Frames: {total_frames}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Fail Frames: {fail_frames_count}", (20, 255), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.putText(result, f"Error Rate: {error_rate:.2f}%", (20, 285), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)

        # 시스템 예외 판단 시 알림 팝업
        if is_failed and not paused:
            cv2.rectangle(result, (400, 300), (880, 420), (0, 0, 255), -1)
            cv2.putText(result, "SYSTEM WARNING", (440, 350), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
            cv2.putText(result, "LANE DETECTION FAILED", (420, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Lane Detection", result)
        cv2.imshow("Edge", edges)
        cv2.imshow("ROI", roi)

        key = cv2.waitKey(30) & 0xFF

        if key == 27: 
            break
        elif key == 32: 
            paused = not paused
        elif key == ord('1'): weather_mode = 'normal'
        elif key == ord('2'): weather_mode = 'fog'
        elif key == ord('3'): weather_mode = 'rain'
        elif key == ord('4'): filter_mode = 'none'
        elif key == ord('5'): filter_mode = 'total'
        elif key == ord('6'): filter_mode = 'fog_spec'
        elif key == ord('7'): filter_mode = 'rain_spec'

    cap.release()
    cv2.destroyAllWindows()

    # 데이터 로깅 시스템
    if total_frames > 0:
        final_accuracy = ((total_frames - fail_frames_count) / total_frames) * 100
        final_error_rate = (fail_frames_count / total_frames) * 100
        current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        log_entry = (
            f"[{current_time}] EXPERIMENT LOG\n"
            f"- 주행 영상 경로  : {VIDEO_PATH}\n"
            f"- 설정 기상 조건  : {weather_mode.upper()}\n"
            f"- 적용 필터 모드  : {filter_mode.upper()}\n"
            f"- 총 분석 프레임  : {total_frames} frames\n"
            f"- 인지 실패 프레임: {fail_frames_count} frames\n"
            f"- 알고리즘 정확도 : {final_accuracy:.2f} %\n"
            f"- 알고리즘 오류율 : {final_error_rate:.2f} %\n"
            f"{'-'*50}\n"
        )
        
        try:
            with open(LOG_FILE_PATH, "a", encoding="utf-8") as f:
                f.write(log_entry)
            print(f"\n💾 실험 데이터 저장 완료 -> {os.path.abspath(LOG_FILE_PATH)}")
        except Exception as e:
            print(f"\n❌ 파일 저장 실패: {e}")
            
        print(log_entry)


if __name__ == "__main__":
    main()
