import cv2
import numpy as np
import matplotlib.pyplot as plt

# 클릭 위치를 저장하기 위한 전역 변수
click_pos = None

def on_mouse_click(event, x, y, flags, param):
    global click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)

def draw_mask_on_background(background, mask_size=(32, 32), num_objects=2):
    global click_pos
    masks = []

    # 윈도우 이름 지정 및 마우스 콜백 함수 설정
    cv2.namedWindow("Select Object Positions on Background (Press Enter to confirm, Esc to exit)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Object Positions on Background (Press Enter to confirm, Esc to exit)", on_mouse_click)

    while True:
        clone = background.copy()

        # 현재 클릭 위치에 마스크 박스 그리기
        if click_pos is not None:
            for i in range(num_objects):
                mask_pos = np.array(click_pos) - np.array(mask_size) // 2 + np.array([i * 50, 0])
                cv2.rectangle(clone, (int(mask_pos[0]), int(mask_pos[1])),
                              (int(mask_pos[0] + mask_size[0]), int(mask_pos[1] + mask_size[1])), (0, 255, 0), 2)

        cv2.imshow("Select Object Positions on Background (Press Enter to confirm, Esc to exit)", clone)

        key = cv2.waitKey(1)
        if key == 27:  # Esc 키를 눌러 종료
            break
        elif key == 13:  # Enter 키를 눌러 확인
            for i in range(num_objects):
                mask = np.zeros(background.shape[:2], dtype=np.uint8)
                mask_pos = np.array(click_pos) - np.array(mask_size) // 2 + np.array([i * 50, 0])
                mask[int(mask_pos[1]):int(mask_pos[1]+mask_size[1]), int(mask_pos[0]):int(mask_pos[0]+mask_size[0])] = 255
                masks.append(mask)

            # 다음 선택을 위해 클릭 위치 초기화
            click_pos = None

    cv2.destroyAllWindows()
    return masks


def resize_and_insert_object(background, object_img, mask):
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if contours are found
    if not contours:
        print("Error: No contours found in the mask.")
        return None

    # Iterate over each contour and insert the object
    result = background.copy()
    for cnt in contours:
        # Find the bounding box of the object
        x, y, w, h = cv2.boundingRect(cnt)

        # Check if the bounding box dimensions are valid
        if w > 0 and h > 0:
            # Resize the object to match the bounding box size
            object_resized = cv2.resize(object_img, (w, h), interpolation=cv2.INTER_AREA)

            # Insert the resized object into the background
            result[y:y+h, x:x+w] = object_resized

    return result

def remove_and_paste_object(input_image_path, background_image_path, output_image_path):
    # 입력 이미지와 배경 이미지를 읽어옵니다.
    image = cv2.imread(input_image_path)
    background = cv2.imread(background_image_path)

    # 이미지를 읽어오지 못했을 경우 예외 처리
    if image is None or background is None:
        print(f"Error: Unable to read the image or background at {input_image_path} or {background_image_path}")
        return

    # 초기 마스크 생성
    mask = np.zeros(image.shape[:2], np.uint8)

    # ROI (Region of Interest) 설정 - 전경에 해당하는 부분을 1로 마스킹
    rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)
    cv2.grabCut(image, mask, rect, None, None, 5, cv2.GC_INIT_WITH_RECT)

    # 마스크를 수정하여 전경을 1로, 배경을 0으로 설정
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 이미지에 새로운 마스크를 곱하여 배경을 제거
    object_no_background = image * mask2[:, :, np.newaxis]

    # 제거된 검은 배경 주변을 약간 흐리게 만들어 줍니다.
    object_no_background = cv2.GaussianBlur(object_no_background, (5, 5), 0)

    # 배경 이미지에서 마스크 그리기
    masks = draw_mask_on_background(background)

    # 초기 마스크 생성
    mask = np.zeros(background.shape[:2], np.uint8)

    # 그린 마스크를 합치기
    for m in masks:
        mask = cv2.bitwise_or(mask, m)

    # 객체 이미지 크기 조정 및 배경에 삽입
    result = resize_and_insert_object(background, object_no_background, mask)

    if result is not None:
        # 배경이 검은색인 부분을 원래 배경 이미지로 채우기
        black_background_mask = (result.sum(axis=2) == 0)
        result[black_background_mask] = background[black_background_mask]

        # 결과 이미지 저장
        cv2.imwrite(output_image_path, result)

    # 시각화를 위한 코드
    plt.subplot(131), plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB)), plt.title('Background Image')
    plt.subplot(132), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Object Image')
    plt.subplot(133), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Result Image with Object Pasted')
    plt.show()

# 예제 사용법
background_image_path = r'C:\Users\sumin\Desktop\inpainting\image\background\P0011.png'  # 입력 이미지 파일 경로
input_image_path = r'C:\Users\sumin\Desktop\inpainting\image\object\car_4.png'  # 배경 이미지 파일 경로
output_image_path = r'C:\Users\sumin\Desktop\inpainting\image\output\result8.png'  # 결과 이미지 파일 경로
remove_and_paste_object(input_image_path, background_image_path, output_image_path)