import cv2
import numpy as np
import matplotlib.pyplot as plt

def draw_mask_on_background(background):
    masks = []
    while True:
        # Display the background image and allow the user to draw a rectangle for the mask
        clone = background.copy()
        r = cv2.selectROI("Select Mask Region on Background (Press Enter to confirm, Esc to exit)", clone)
        if r[2] == 0 or r[3] == 0:  # Check if the user canceled the selection
            break
        mask = np.zeros(background.shape[:2], dtype=np.uint8)  # Ensure single-channel mask
        mask[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])] = 255
        masks.append(mask)
    cv2.destroyAllWindows()
    return masks

def resize_and_insert_object(background, object_img, mask):
    # 마스크에서 컨투어 찾기
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 컨투어를 찾았는지 확인
    if not contours:
        print("에러: 마스크에서 컨투어를 찾을 수 없습니다.")
        return None

    # 각 컨투어에 대해 객체 삽입 수행
    result = background.copy()
    for cnt in contours:
        # 객체의 경계 상자 찾기
        x, y, w, h = cv2.boundingRect(cnt)

        # 경계 상자의 크기가 유효한지 확인
        if w > 0 and h > 0:
            # 경계 상자의 크기에 맞게 객체 크기 조정
            object_resized = cv2.resize(object_img, (w, h), interpolation=cv2.INTER_AREA)

            # 조정된 객체를 배경에 삽입
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
        
        # 시각화를 위한 코드s
        plt.subplot(131), plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB)), plt.title('Background Image')
        plt.subplot(132), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Object Image')
        plt.subplot(133), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Result Image with Object Pasted')
        plt.show()

# 예제 사용법
background_image_path = r'C:\Users\sumin\Desktop\inpainting\image\background\P0054.png'  # 입력 이미지 파일 경로
input_image_path = r'C:\Users\sumin\Desktop\inpainting\image\object\car_4.png'  # 배경 이미지 파일 경로
output_image_path = r'C:\Users\sumin\Desktop\inpainting\image\output\result3.png'  # 결과 이미지 파일 경로
remove_and_paste_object(input_image_path, background_image_path, output_image_path)

