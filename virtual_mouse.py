import cv2
import mediapipe as mp
import pyautogui
import time

# ==============================
# SETTINGS
# ==============================

smoothening = 5
scroll_speed = 60
scroll_threshold = 8
click_cooldown = 1

frame_reduction = 80

screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)

prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

last_click_time = 0
prev_middle_y = 0

cap = cv2.VideoCapture(0)


# ==============================
# FUNCTION: Detect fingers up
# ==============================

def fingers_up(lm):
    fingers = []

    # Thumb
    if lm[4].x > lm[3].x:
        fingers.append(1)
    else:
        fingers.append(0)

    # Index
    if lm[8].y < lm[6].y:
        fingers.append(1)
    else:
        fingers.append(0)

    # Middle
    if lm[12].y < lm[10].y:
        fingers.append(1)
    else:
        fingers.append(0)

    return fingers


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    mode_text = "MOVE MODE"

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            lm = hand_landmarks.landmark

            index = lm[8]
            middle = lm[12]
            thumb = lm[4]

            index_x = int(index.x * frame_w)
            index_y = int(index.y * frame_h)

            middle_x = int(middle.x * frame_w)
            middle_y = int(middle.y * frame_h)

            thumb_x = int(thumb.x * frame_w)
            thumb_y = int(thumb.y * frame_h)

            cv2.circle(frame, (index_x, index_y), 10, (0,255,0), -1)
            cv2.circle(frame, (middle_x, middle_y), 10, (255,255,0), -1)
            cv2.circle(frame, (thumb_x, thumb_y), 10, (255,0,0), -1)

            fingers = fingers_up(lm)

            # ==============================
            # SCROLL MODE (Two fingers up)
            # ==============================

            if fingers[1] == 1 and fingers[2] == 1:

                mode_text = "SCROLL MODE"

                # Detect vertical movement
                diff = middle_y - prev_middle_y

                if abs(diff) > scroll_threshold:

                    if diff < 0:
                        pyautogui.scroll(scroll_speed)   # Scroll UP
                    else:
                        pyautogui.scroll(-scroll_speed)  # Scroll DOWN

                prev_middle_y = middle_y

            # ==============================
            # MOVE MODE (One finger)
            # ==============================

            else:

                screen_x = int(
                    (index_x - frame_reduction)
                    * screen_w
                    / (frame_w - 2 * frame_reduction)
                )

                screen_y = int(
                    (index_y - frame_reduction)
                    * screen_h
                    / (frame_h - 2 * frame_reduction)
                )

                curr_x = prev_x + (screen_x - prev_x) / smoothening
                curr_y = prev_y + (screen_y - prev_y) / smoothening

                pyautogui.moveTo(curr_x, curr_y)

                prev_x, prev_y = curr_x, curr_y

                prev_middle_y = middle_y  # reset reference

            current_time = time.time()

            # ==============================
            # LEFT CLICK
            # ==============================

            if abs(index_x - thumb_x) < 30 and abs(index_y - thumb_y) < 30:
                if current_time - last_click_time > click_cooldown:
                    pyautogui.click()
                    last_click_time = current_time
                    mode_text = "LEFT CLICK"

            # ==============================
            # RIGHT CLICK
            # ==============================

            if abs(index_x - middle_x) < 40 and abs(index_y - middle_y) < 40:
                if current_time - last_click_time > click_cooldown:
                    pyautogui.rightClick()
                    last_click_time = current_time
                    mode_text = "RIGHT CLICK"

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, mode_text, (50, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1,
                (0,255,255), 2)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
