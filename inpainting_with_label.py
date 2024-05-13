import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# 클래스 레이블 및 상수 정의
CLASS_LABELS = ['label1', 'label2']
LABELS_FOLDER_PATH = ""  # 라벨 폴더 경로
OBJECT_IMAGES_FOLDER = "" # 삽입할 객체 이미지 폴더 경로
BACKGROUND_IMAGES_FOLDER = "" # 배경 이미지 폴더 경로
OUTPUT_IMAGES_FOLDER = "" # 결과 이미지 저장할 폴더 경로

# 클릭 위치 및 객체 수를 저장하기 위한 전역 변수
click_pos = None
object_count = 0
img_width = 0
img_height = 0
label_path = ""

def draw_mask_on_background(background, mask_size=(32, 32), second_object=False):
    global click_pos, object_count, label_path
    masks = []

    window_name = "Select Object Position on Background"
    if second_object:
        window_name += " (Second Object)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse_click_with_label, (background, label_path))

    while True:
        clone = background.copy()

        # 이전에 선택된 마스크 그리기
        for mask in masks:
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                cv2.rectangle(clone, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 현재 클릭 위치에 상자 그리기
        if click_pos is not None:
            mask_pos = np.array(click_pos) - np.array(mask_size) // 2
            cv2.rectangle(clone, (int(mask_pos[0]), int(mask_pos[1])),
                          (int(mask_pos[0] + mask_size[0]), int(mask_pos[1] + mask_size[1])), (0, 255, 0), 2)

        cv2.imshow(window_name, clone)

        key = cv2.waitKey(1)
        if key == 27:  # 종료하려면 Esc 키를 누름
            break
        elif key == 13:  # 확인하려면 Enter 키를 누름
            if click_pos is not None:
                # 지정된 위치에 고정 크기의 마스크 생성
                mask = np.zeros(background.shape[:2], dtype=np.uint8)
                mask_pos = np.array(click_pos) - np.array(mask_size) // 2
                mask[int(mask_pos[1]):int(mask_pos[1]+mask_size[1]), int(mask_pos[0]):int(mask_pos[0]+mask_size[0])] = 255

                # 객체의 수에 따라 마스크의 색 설정
                mask_color = 255 if object_count == 0 else 128
                mask[mask == 255] = mask_color

                masks.append(mask)

                # 라벨 추가
                if label_path:
                    write_label(label_path, CLASS_LABELS[object_count], mask_pos[0], mask_pos[1], mask_size[0], mask_size[1], img_width, img_height)

                # 다음 선택을 위해 클릭 위치 재설정
                click_pos = None

                # 객체 수 증가
                object_count += 1

    cv2.destroyAllWindows()
    return masks

def resize_and_insert_object(background, object_img, mask, position):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print("Error: No contours found in the mask.")
        return None

    result = background.copy()
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 0 and h > 0:
            object_resized = cv2.resize(object_img, (w, h), interpolation=cv2.INTER_AREA)
            result[y:y+h, x:x+w] = object_resized

    return result

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

def insert_object_with_grabcut(background, object_img, mask):
    bgd_model = np.zeros((1, 65), dtype=np.float64)
    fgd_model = np.zeros((1, 65), dtype=np.float64)

    mask, _, _ = cv2.grabCut(object_img, mask, None, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)

    result_mask = np.where((mask == 1) | (mask == 3), 255, 0).astype('uint8')
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

def generate_label(class_label, x, y, width, height, img_width, img_height):
    relative_x = x / img_width + 0.003
    relative_y = y / img_height + 0.003
    relative_width = width / img_width
    relative_height = height / img_height
    return f"{CLASS_LABELS.index(class_label)} {relative_x} {relative_y} {relative_width} {relative_height}"

def write_label(label_path, class_label, x, y, width, height, img_width, img_height):
    # Calculate relative positions and sizes
    relative_x = x / img_width + 0.003
    relative_y = y / img_height + 0.003
    relative_width = width / img_width
    relative_height = height / img_height

    # Create the label string
    label_string = f"{CLASS_LABELS.index(class_label)} {relative_x} {relative_y} {relative_width} {relative_height}\n"

    # Write the label to file
    try:
        with open(label_path, 'a') as file:  # Append mode
            file.write(label_string)
    except IOError as error:
        print(f"Error writing to label file {label_path}: {error}")

def on_mouse_click_with_label(event, x, y, flags, param):
    global click_pos, object_count, img_width, img_height, label_path
    if event == cv2.EVENT_LBUTTONDOWN:
        click_pos = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN: # 마우스 오른쪽 클릭시 마지막으로 찍은 객체 삭제
        object_count -= 1
        click_pos = None
        img_width, img_height, _ = param[0].shape
        label_path = param[1]

def select_object_image_with_label(object_images_folder):
    print("Select an object image to insert:")
    for i, filename in enumerate(os.listdir(object_images_folder)):
        print(f"{i + 1}. {filename}")
    while True:
        try:
            choice = int(input("Enter the number of the object image to insert: ")) - 1
            if 0 <= choice < len(os.listdir(object_images_folder)):
                # 사용자가 선택한 인덱스를 그대로 object_class로 사용
                return choice
            else:
                print("Invalid choice. Please enter a valid number.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def draw_mask_on_background_with_label(background, mask, label_path):
    global click_pos, object_count, img_width, img_height

    img_height, img_width, _ = background.shape

    # 창 설정
    window_name = "Select Object Position on Background (Press Enter to confirm, Esc to exit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, on_mouse_click_with_label, background)

    while True:
        clone = background.copy()

        # 현재 클릭 위치에 상자 그리기
        if click_pos is not None:
            mask_pos = np.array(click_pos) - np.array((16, 16))
            cv2.rectangle(clone, (int(mask_pos[0]), int(mask_pos[1])), 
                          (int(mask_pos[0] + 32), int(mask_pos[1] + 32)), (0, 255, 0), 2)

        cv2.imshow(window_name, clone)

        key = cv2.waitKey(1)
        if key == 27:  # 종료
            break
        elif key == 13:  # 확인
            if click_pos is not None:
                # 마스크 업데이트
                mask[int(mask_pos[1]):int(mask_pos[1] + 32), int(mask_pos[0]):int(mask_pos[0] + 32)] = 255

                # 라벨 추가
                if label_path:
                    write_label(label_path, CLASS_LABELS[object_count], mask_pos[0], mask_pos[1], 32, 32, img_width, img_height)

                # 다음 객체를 위해 변수 업데이트
                click_pos = None
                object_count += 1

    cv2.destroyAllWindows()

    return mask

def resize_and_insert_object(background, object_img, mask):
    # Example implementation that returns the modified image and a bounding box
    # Assuming mask is a binary mask where the object should be inserted

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assuming we take the first contour found for this example
        x, y, w, h = cv2.boundingRect(contours[0])
        # Resize the object image to fit the bounding box
        object_resized = cv2.resize(object_img, (w, h), interpolation=cv2.INTER_AREA)
        # Insert the resized object into the background image
        result = background.copy()
        result[y:y+h, x:x+w] = object_resized

        # The bounding box is returned as a tuple (x, y, w, h)
        return result, (x, y, w, h)
    else:
        # If no contours were found, return the original image and None for the bounding box
        return background, None

def apply_grabcut(image):
    """
    GrabCut 알고리즘을 사용하여 이미지에서 배경을 제거합니다.
    :param image: 배경을 제거할 객체 이미지
    :return: 배경이 제거된 이미지와 마스크
    """
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (50, 50, image.shape[1] - 100, image.shape[0] - 100)  # 객체가 위치할 대략적인 사각형 지정
    
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    image_foreground = image * mask2[:, :, np.newaxis]  # 배경이 제거된 이미지
    
    return image_foreground, mask2

def resize_and_insert_object(background, object_img, mask):
    # Example implementation that returns the modified image and a bounding box
    # Assuming mask is a binary mask where the object should be inserted

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assuming we take the first contour found for this example
        x, y, w, h = cv2.boundingRect(contours[0])
        # Resize the object image to fit the bounding box
        object_resized = cv2.resize(object_img, (w, h), interpolation=cv2.INTER_AREA)
        # Insert the resized object into the background image
        result = background.copy()
        result[y:y+h, x:x+w] = object_resized

        black_background_mask = np.all(result == [0, 0, 0], axis=-1)
        result[black_background_mask] = background[black_background_mask]

        # The bounding box is returned as a tuple (x, y, w, h)
        return result, (x, y, w, h)
    else:
        # If no contours were found, return the original image and None for the bounding box
        return background, None

def remove_and_paste_object_with_label(object_image_path, background_image_path, output_image_path, label_path, object_class):
    global img_width, img_height

    object_img = cv2.imread(object_image_path)
    background_img = cv2.imread(background_image_path)
    
    if background_img is None or object_img is None:
        print(f"Error: Unable to read the images at {object_image_path} or {background_image_path}")
        return

    img_height, img_width, _ = background_img.shape

    object_img_foreground, mask = apply_grabcut(object_img)  # 배경 제거된 객체 이미지와 마스크를 얻음

    # 현재 배경 이미지의 업데이트된 복사본을 저장할 변수
    updated_background_img = background_img.copy()

    masks = draw_mask_on_background(background_img)  # 사용자가 객체를 삽입할 위치를 선택
    for insert_mask in masks:
        # 삽입 위치에 따라 객체를 업데이트된 배경 이미지에 삽입
        result, bounding_box = resize_and_insert_object(updated_background_img, object_img_foreground, insert_mask)
        if bounding_box is not None:
            x, y, w, h = bounding_box
            write_label(label_path, CLASS_LABELS[object_class], x, y, w, h, img_width, img_height)
            updated_background_img = result.copy()  # 객체가 삽입된 이미지로 배경 이미지 업데이트

    # 최종 결과 이미지 저장
    if result is not None:
        cv2.imwrite(output_image_path, result)
    else:
        print("Error occurred during the object insertion process.")

    return mask

def remove_and_paste_objects_with_label(input_images_folder, background_images_folder, output_images_folder):
    global object_count  # object_count를 전역 변수로 사용하기 위해 추가
    global img_width, img_height  # img_width와 img_height를 전역 변수로 사용하기 위해 추가

    # remove_and_paste_objects_with_label 함수 내에서 수정
    for background_filename in os.listdir(background_images_folder):
        object_count = 0  # 객체 수 초기화
        if background_filename.endswith('.png') or background_filename.endswith('.jpg'):
            background_image_path = os.path.join(background_images_folder, background_filename)
            object_class = select_object_image_with_label(input_images_folder)  # object_class 값을 받음
            if object_class is None:  # 이전 예제 코드에서는 object_image_path를 반환했으나, 이제는 object_class를 직접 사용
                print("Error: No object image selected.")
                continue

            object_image_path = os.path.join(input_images_folder, os.listdir(input_images_folder)[object_class])  # object_class에 해당하는 실제 경로를 얻음
            output_image_path = os.path.join(output_images_folder, background_filename)
            label_path = os.path.join(LABELS_FOLDER_PATH, os.path.splitext(background_filename)[0] + '.txt')

            # 수정된 부분: remove_and_paste_object_with_label 함수에 object_class 인자를 추가하여 전달
            mask = remove_and_paste_object_with_label(object_image_path, background_image_path, output_image_path, label_path, object_class)

            if mask is not None:
                print(f"Objects successfully removed and pasted on {background_filename}.")
            else:
                print(f"Error occurred during processing of {background_filename}.")

remove_and_paste_objects_with_label(OBJECT_IMAGES_FOLDER, BACKGROUND_IMAGES_FOLDER, OUTPUT_IMAGES_FOLDER)
