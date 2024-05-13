import cv2
import numpy as np
import matplotlib.pyplot as plt

# 클릭 위치를 저장하기 위한 전역 변수
click_pos = None

def on_mouse_click(event, x, y, flags, param):
    global click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)

def draw_mask_on_background(background, mask_size=(32, 32)):
    global click_pos
    masks = []

    # 창을 명시적으로 생성합니다
    cv2.namedWindow("Select Object Position on Background (Press Enter to confirm, Esc to exit)", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Select Object Position on Background (Press Enter to confirm, Esc to exit)", on_mouse_click)

    while True:
        # 배경 이미지를 표시합니다
        clone = background.copy()

        # 이전에 선택된 마스크를 그립니다
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 현재 클릭 위치에 상자를 그립니다
        if click_pos is not None:
            mask_pos = np.array(click_pos) - np.array(mask_size) // 2
            cv2.rectangle(clone, (int(mask_pos[0]), int(mask_pos[1])),
                          (int(mask_pos[0] + mask_size[0]), int(mask_pos[1] + mask_size[1])), (0, 255, 0), 2)

        cv2.imshow("Select Object Position on Background (Press Enter to confirm, Esc to exit)", clone)

        key = cv2.waitKey(1)
        if key == 27:  # 종료하려면 Esc 키를 누릅니다
            break
        elif key == 13:  # 확인하려면 Enter 키를 누릅니다
            if click_pos is not None:
                # 지정된 위치에 고정 크기의 마스크를 생성합니다
                mask = np.zeros(background.shape[:2], dtype=np.uint8)
                mask_pos = np.array(click_pos) - np.array(mask_size) // 2
                mask[int(mask_pos[1]):int(mask_pos[1]+mask_size[1]), int(mask_pos[0]):int(mask_pos[0]+mask_size[0])] = 255

                masks.append(mask)

                # 다음 선택을 위해 클릭 위치를 재설정합니다
                click_pos = None

    cv2.destroyAllWindows()
    return masks

def resize_and_insert_object(background, object_img, mask):
    # 마스크에서 윤곽선을 찾습니다
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선을 찾았는지 확인합니다
    if not contours:
        print("Error: No contours found in the mask.")
        return None

    # 각 윤곽선을 순회하고 객체를 삽입합니다
    result = background.copy()
    for cnt in contours:
        # 객체의 경계 상자를 찾습니다
        x, y, w, h = cv2.boundingRect(cnt)

        # 경계 상자의 크기가 유효한지 확인합니다
        if w > 0 and h > 0:
            # 객체를 경계 상자 크기에 맞게 조정합니다
            object_resized = cv2.resize(object_img, (w, h), interpolation=cv2.INTER_AREA)

            # 조정된 객체를 배경에 삽입합니다
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
background_image_path = r'C:\Users\sumin\Desktop\Augmentation\train_LnS\P2335.png'  # 입력 이미지 파일 경로
input_image_path = r'C:\Users\sumin\Desktop\Augmentation\object\ship.png'  # 배경 이미지 파일 경로
output_image_path = r'C:\Users\sumin\Desktop\Augmentation\Aug_train_LnS\P2335.png'  # 결과 이미지 파일 경로
remove_and_paste_object(input_image_path, background_image_path, output_image_path)