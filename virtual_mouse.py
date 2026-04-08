import cv2
import mediapipe as mp
import pyautogui
import time

screen_w, screen_h = pyautogui.size()

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(max_num_hands=1)

smoothening = 5
prev_x, prev_y = 0, 0
curr_x, curr_y = 0, 0

click_cooldown = 1
last_click_time = 0

cap = cv2.VideoCapture(0)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    frame_h, frame_w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

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

            # Cursor movement
            screen_x = int(index.x * screen_w)
            screen_y = int(index.y * screen_h)

            curr_x = prev_x + (screen_x - prev_x) / smoothening
            curr_y = prev_y + (screen_y - prev_y) / smoothening

            pyautogui.moveTo(curr_x, curr_y)

            prev_x, prev_y = curr_x, curr_y

            current_time = time.time()

            # Left Click
            if abs(index_x - thumb_x) < 30 and abs(index_y - thumb_y) < 30:
                if current_time - last_click_time > click_cooldown:
                    pyautogui.click()
                    last_click_time = current_time
                    cv2.putText(frame,"LEFT CLICK",(50,50),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

            # Right Click
            if abs(index_x - middle_x) < 40 and abs(index_y - middle_y) < 40:
                if current_time - last_click_time > click_cooldown:
                    pyautogui.rightClick()
                    last_click_time = current_time
                    cv2.putText(frame,"RIGHT CLICK",(50,100),
                                cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2)

            # Scroll gesture
            if abs(middle_y - index_y) > 100:
                pyautogui.scroll(20)
                cv2.putText(frame,"SCROLL",(50,150),
                            cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,255),2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Virtual Mouse", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()