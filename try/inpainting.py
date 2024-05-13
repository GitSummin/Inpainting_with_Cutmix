import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 클릭 위치를 저장하기 위한 전역 변수
click_pos = None
# 객체 수를 저장하기 위한 전역 변수
object_count = 0

def on_mouse_click(event, x, y, flags, param):
    global click_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)

def draw_mask_on_background(background, mask_size=(32, 32), second_object=False):
    global click_pos, object_count
    masks = []

    # 창을 명시적으로 생성합니다
    if second_object:
        window_name = "Select Second Object Position on Background (Press Enter to confirm, Esc to exit)"
    else:
        window_name = "Select Object Position on Background (Press Enter to confirm, Esc to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse_click)

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

        cv2.imshow(window_name, clone)

        key = cv2.waitKey(1)
        if key == 27:  # 종료하려면 Esc 키를 누릅니다
            break
        elif key == 13:  # 확인하려면 Enter 키를 누릅니다
            if click_pos is not None:
                # 지정된 위치에 고정 크기의 마스크를 생성합니다
                mask = np.zeros(background.shape[:2], dtype=np.uint8)
                mask_pos = np.array(click_pos) - np.array(mask_size) // 2
                mask[int(mask_pos[1]):int(mask_pos[1]+mask_size[1]), int(mask_pos[0]):int(mask_pos[0]+mask_size[0])] = 255

                # 객체의 수에 따라 마스크의 색을 다르게 설정합니다
                mask_color = 255 if object_count == 0 else 128  # 255와 128로 설정합니다.
                mask[mask == 255] = mask_color

                masks.append(mask)

                # 다음 선택을 위해 클릭 위치를 재설정합니다
                click_pos = None

                # 객체의 수를 증가시킵니다
                object_count += 1

    cv2.destroyAllWindows()
    return masks

def resize_and_insert_object(background, object_img, mask, position):
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

def remove_and_paste_object(input_image_path, background_image_path, output_image_path, second_object_path=None):
    global object_count
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

    # 배경 이미지에서 첫 번째 객체를 삽입합니다.
    masks = draw_mask_on_background(background)

    # 초기 마스크 생성
    mask = np.zeros(background.shape[:2], np.uint8)

    # 그린 마스크를 합치기
    for m in masks:
        mask = cv2.bitwise_or(mask, m)

    # 객체 이미지 크기 조정 및 배경에 삽입
    result = resize_and_insert_object(background, object_no_background, mask, (0, 0))

    if result is not None:
        # 배경이 검은색인 부분을 원래 배경 이미지로 채우기
        black_background_mask = (result.sum(axis=2) == 0)
        result[black_background_mask] = background[black_background_mask]

        # 결과 이미지 저장
        cv2.imwrite(output_image_path, result)

        # 첫 번째 객체 삽입 완료 후, 두 번째 객체 삽입을 위한 과정 시작
        if second_object_path is not None:
            # 결과 이미지를 띄우기
            cv2.imshow("Result with First Object", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 두 번째 객체를 읽어옵니다.
            second_object = cv2.imread(second_object_path)
            if second_object is None:
                print(f"Error: Unable to read the second object image at {second_object_path}")
                return

            # 두 번째 객체를 32x32 크기로 축소합니다.
            second_object_resized = cv2.resize(second_object, (32, 32), interpolation=cv2.INTER_AREA)

            # 두 번째 객체를 배경 이미지에 삽입합니다.
            result_with_second_object = resize_and_insert_object(result, second_object_resized, np.ones((32, 32), dtype=np.uint8)*255, (0, 0))

            if result_with_second_object is not None:
                # 결과 이미지 저장
                cv2.imwrite(output_image_path, result_with_second_object)
            else:
                print("Error occurred during second object insertion.")
    else:
        print("Error occurred during object insertion.")
        
    # 시각화를 위한 코드
    # plt.subplot(131), plt.imshow(cv2.cvtColor(background, cv2.COLOR_BGR2RGB)), plt.title('Background Image')
    # plt.subplot(132), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Object Image')
    # plt.subplot(133), plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB)), plt.title('Result Image with Object Pasted')
    # plt.show()

def insert_object_with_grabcut(background, object_img, mask):
    # GrabCut 알고리즘을 적용하여 배경을 제거한 후, 객체 삽입
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    # GrabCut 적용
    mask, _, _ = cv2.grabCut(object_img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    # 결과 마스크에서 객체 영역(1, 3)을 객체로 처리
    result_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')

    # 객체 이미지 크기 조정 및 배경에 삽입
    result = resize_and_insert_object(background, object_img, result_mask, (0, 0))

    return result

def select_object_image(object_images_folder):
    print("Select an object image to insert:")
    for i, filename in enumerate(os.listdir(object_images_folder)):
        print(f"{i + 1}. {filename}")
    while True:
        try:
            choice = int(input("Enter the number of the object image to insert: ")) - 1
            if choice >= 0 and choice < len(os.listdir(object_images_folder)):
                return os.path.join(object_images_folder, os.listdir(object_images_folder)[choice])
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def remove_and_paste_objects(input_images_folder, background_images_folder, output_images_folder):
    # 배경 이미지 폴더에서 모든 이미지 파일을 읽어옵니다.
    for background_filename in os.listdir(background_images_folder):
        if background_filename.endswith('.png') or background_filename.endswith('.jpg'):
            # 배경 이미지 파일의 전체 경로
            background_image_path = os.path.join(background_images_folder, background_filename)

            # 사용자로부터 객체 이미지를 선택합니다.
            object_image_path = select_object_image(input_images_folder)
            if object_image_path is None:
                print("Error: No object image selected.")
                continue

            # 객체를 배경 이미지에 삽입하여 결과 이미지 생성
            output_image_path = os.path.join(output_images_folder, background_filename)
            remove_and_paste_object(object_image_path, background_image_path, output_image_path)

# 예제 사용법
object_images_folder = r'C:\Users\sumin\Desktop\Augmentation_DOTAv2.0\object'  # 객체 이미지 폴더 경로
background_images_folder = r'C:\Users\sumin\Desktop\TOD_data\DOTA-v2.0\train\images' # 배경 이미지 폴더 경로 
output_images_folder = r'C:\Users\sumin\Desktop\Augmentation_DOTAv2.0\train\images'  # 결과 이미지 저장 폴더 경로

# 함수 호출하여 객체를 제거하고 배경에 삽입한 결과 이미지를 생성합니다.
remove_and_paste_objects(object_images_folder, background_images_folder, output_images_folder)


def remove_and_paste_object(object_image_path, background_image_path, output_image_path, second_object_path=None):
    global object_count, img_width, img_height, label_path
    # 입력 이미지와 배경 이미지를 읽어옵니다.
    image = cv2.imread(object_image_path)
    background = cv2.imread(background_image_path)

    # 이미지를 읽어오지 못했을 경우 예외 처리
    if image is None or background is None:
        print(f"Error: Unable to read the image or background at {object_image_path} or {background_image_path}")
        return None  # None을 반환하여 마스크가 없음을 표시

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

    # 배경 이미지에서 첫 번째 객체를 삽입합니다.
    masks = draw_mask_on_background(background)

    # 초기 마스크 생성
    mask = np.zeros(background.shape[:2], np.uint8)

    # 그린 마스크를 합치기
    for m in masks:
        mask = cv2.bitwise_or(mask, m)

    # 객체 이미지에 그랩컷 적용
    object_mask = apply_grabcut(object_no_background)
    object_masked = cv2.bitwise_and(object_no_background, object_no_background, mask=object_mask)

    # 배경 이미지에 객체 삽입
    masks = draw_mask_on_background(background, second_object=bool(second_object_path))
    for mask in masks:
        background = resize_and_insert_object(background, object_masked, mask, (0, 0))

    # 객체 이미지 크기 조정 및 배경에 삽입
    result = resize_and_insert_object(background, object_no_background, mask, (0, 0))

    if result is not None:
        # 배경이 검은색인 부분을 원래 배경 이미지로 채우기
        black_background_mask = (result.sum(axis=2) == 0)
        result[black_background_mask] = background[black_background_mask]

        # 결과 이미지 저장
        cv2.imwrite(output_image_path, result)

        # 첫 번째 객체 삽입 완료 후, 두 번째 객체 삽입을 위한 과정 시작
        if second_object_path is not None:
            # 결과 이미지를 띄우기
            cv2.imshow("Result with First Object", result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            # 두 번째 객체를 읽어옵니다.
            second_object = cv2.imread(second_object_path)
            if second_object is None:
                print(f"Error: Unable to read the second object image at {second_object_path}")
                return

            # 두 번째 객체를 32x32 크기로 축소합니다.
            second_object_resized = cv2.resize(second_object, (64, 64), interpolation=cv2.INTER_AREA)

            # 두 번째 객체를 배경 이미지에 삽입합니다.
            result_with_second_object = resize_and_insert_object(result, second_object_resized, np.ones((32, 32), dtype=np.uint8)*255, (0, 0))

            if result_with_second_object is not None:
                # 결과 이미지 저장
                cv2.imwrite(output_image_path, result_with_second_object)
            else:
                print("Error occurred during second object insertion.")
    else:
        print("Error occurred during object insertion.")
    
    return mask