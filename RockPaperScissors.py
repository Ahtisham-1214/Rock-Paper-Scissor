import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import random
import logging
from datetime import datetime
import os

MODEL_PATH = 'best_model.keras'
LOGS_DIR = "logs"
CONFIDENCE_THRESHOLD = 0.70
MAX_ROUNDS = 3


CLASS_NAMES = ['paper', 'rock', 'scissors']


if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
    print(f"Created directory: {LOGS_DIR}")

log_file_path = os.path.join(LOGS_DIR, 'game_log.txt')
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s', filemode='a')

try:
    model = load_model(MODEL_PATH)
    logging.info(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Please ensure '{MODEL_PATH}' exists and was trained correctly.")
    logging.error(f"Error loading model '{MODEL_PATH}': {e}")
    exit()

def get_computer_move():
    return random.choice(CLASS_NAMES)

def get_winner(player_move, computer_move):
    if player_move == "unknown":
        return "Unknown - Retry"
    if player_move == computer_move:
        return "Draw"
    elif (player_move == "rock" and computer_move == "scissors") or \
         (player_move == "scissors" and computer_move == "paper") or \
         (player_move == "paper" and computer_move == "rock"):
        return "Player"
    else:
        return "Computer"

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    logging.error("Could not open webcam.")
    exit()

cv2.namedWindow("Rock Paper Scissors Game", cv2.WINDOW_NORMAL)

player_score = 0
computer_score = 0
round_number = 1
round_over = False
game_over = False
winner_text = ""
player_move_display = ""
computer_move_display = ""

logging.info("Game started.")

# --- Main Game Loop ---
while True:
    ret, frame = cap.read()
    if not ret:
        logging.error("Failed to grab frame from webcam")
        break

    frame = cv2.flip(frame, 1)
    display_frame = frame.copy()
    h, w, _ = display_frame.shape

    roi_size = 250
    x1 = w // 2 - roi_size // 2
    y1 = h // 2 - roi_size // 2 - 50
    x2 = x1 + roi_size
    y2 = y1 + roi_size

    text_color = (255, 255, 255)
    line_type = cv2.LINE_AA
    font_scale_large = 1
    font_scale_medium = 0.8
    font_scale_small = 0.6

    if game_over:
        final_msg = "Game Over!"
        if player_score > computer_score: final_msg += " You Won the Game!"
        elif computer_score > player_score: final_msg += " Computer Won the Game!"
        else: final_msg += " It's a Tie Game!"

        # Center text based on text length
        final_msg_size = cv2.getTextSize(final_msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, 2)[0]
        final_msg_x = (w - final_msg_size[0]) // 2

        score_msg = f'Final Score: Player {player_score} - CPU {computer_score}'
        score_msg_size = cv2.getTextSize(score_msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, 2)[0]
        score_msg_x = (w - score_msg_size[0]) // 2

        instruction_msg = 'Press Space to Play Again or Q to Quit'
        instruction_msg_size = cv2.getTextSize(instruction_msg, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, 2)[0]
        instruction_msg_x = (w - instruction_msg_size[0]) // 2

        # Draw text with background for better visibility
        cv2.rectangle(display_frame, (final_msg_x - 10, h // 2 - 70), (final_msg_x + final_msg_size[0] + 10, h // 2 - 30), (0, 0, 0), -1)
        cv2.putText(display_frame, final_msg, (final_msg_x, h // 2 - 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, (0, 255, 255), 2, line_type)

        cv2.rectangle(display_frame, (score_msg_x - 10, h // 2 - 10), (score_msg_x + score_msg_size[0] + 10, h // 2 + 30), (0, 0, 0), -1)
        cv2.putText(display_frame, score_msg, (score_msg_x, h // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, text_color, 2, line_type)

        cv2.rectangle(display_frame, (instruction_msg_x - 10, h // 2 + 40), (instruction_msg_x + instruction_msg_size[0] + 10, h // 2 + 80), (0, 0, 0), -1)
        cv2.putText(display_frame, instruction_msg, (instruction_msg_x, h // 2 + 70), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, text_color, 2, line_type)

    elif round_over:
        current_y = h // 2 - 120
        line_spacing = 40  # Increased spacing for better readability

        win_color = (0, 255, 0) if "Player" in winner_text else (0,0,255) if "Computer" in winner_text else (255,255,0) if "Draw" in winner_text else text_color

        # Center each text line
        winner_size = cv2.getTextSize(winner_text, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)[0]
        winner_x = (w - winner_size[0]) // 2

        player_text = f'Player: {player_move_display}'
        player_size = cv2.getTextSize(player_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, 2)[0]
        player_x = (w - player_size[0]) // 2

        computer_text = f'Computer: {computer_move_display}'
        computer_size = cv2.getTextSize(computer_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, 2)[0]
        computer_x = (w - computer_size[0]) // 2

        score_text = f'Score: Player {player_score} - CPU {computer_score}'
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, 2)[0]
        score_x = (w - score_size[0]) // 2

        next_text = 'Press Space for Final Results' if round_number >= MAX_ROUNDS else 'Press Space for Next Round'
        next_size = cv2.getTextSize(next_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, 2)[0]
        next_x = (w - next_size[0]) // 2

        # Draw text with background rectangles for better visibility
        cv2.rectangle(display_frame, (winner_x - 10, current_y - 30), (winner_x + winner_size[0] + 10, current_y + 10), (0, 0, 0), -1)
        cv2.putText(display_frame, winner_text, (winner_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, 1.1, win_color, 2, line_type)
        current_y += line_spacing + 10

        cv2.rectangle(display_frame, (player_x - 10, current_y - 30), (player_x + player_size[0] + 10, current_y + 10), (0, 0, 0), -1)
        cv2.putText(display_frame, player_text, (player_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, (0, 255, 255), 2, line_type)
        current_y += line_spacing

        cv2.rectangle(display_frame, (computer_x - 10, current_y - 30), (computer_x + computer_size[0] + 10, current_y + 10), (0, 0, 0), -1)
        cv2.putText(display_frame, computer_text, (computer_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, (0, 255, 255), 2, line_type)
        current_y += line_spacing

        cv2.rectangle(display_frame, (score_x - 10, current_y - 30), (score_x + score_size[0] + 10, current_y + 10), (0, 0, 0), -1)
        cv2.putText(display_frame, score_text, (score_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, text_color, 2, line_type)
        current_y += line_spacing

        cv2.rectangle(display_frame, (next_x - 10, current_y - 30), (next_x + next_size[0] + 10, current_y + 10), (0, 0, 0), -1)
        cv2.putText(display_frame, next_text, (next_x, current_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, text_color, 2, line_type)

    elif not game_over and not round_over:
        # Top header with round and score info
        round_text = f'Round {round_number}/{MAX_ROUNDS}'
        round_size = cv2.getTextSize(round_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, 2)[0]

        score_text = f'Score: Player {player_score} - CPU {computer_score}'
        score_size = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, 2)[0]

        # Position round text to left, score text remains centered
        round_x = 20  # Fixed position from left edge
        score_x = 20

        # Draw background rectangles for better visibility
        cv2.rectangle(display_frame, (round_x - 10, 20), (round_x + round_size[0] + 10, 60), (0, 0, 0), -1)
        cv2.putText(display_frame, round_text, (round_x, 50), cv2.FONT_HERSHEY_SIMPLEX, font_scale_large, text_color, 2, line_type)

        cv2.rectangle(display_frame, (score_x - 10, 70), (score_x + score_size[0] + 10, 110), (0, 0, 0), -1)
        cv2.putText(display_frame, score_text, (score_x, 100), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, text_color, 2, line_type)

        # Instruction for hand placement - centered below the box
        place_text = 'Place hand in Green Box'
        place_size = cv2.getTextSize(place_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, 2)[0]
        place_x = x1 + (roi_size - place_size[0]) // 2

        cv2.rectangle(display_frame, (place_x - 5, y2 + 5), (place_x + place_size[0] + 5, y2 + 30), (0, 0, 0), -1)
        cv2.putText(display_frame, place_text, (place_x, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, font_scale_small, (0, 255, 0), 2, line_type)

        # Bottom instruction
        play_text = 'Press Space to Play'
        play_size = cv2.getTextSize(play_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, 2)[0]
        play_x = (w - play_size[0]) // 2

        cv2.rectangle(display_frame, (play_x - 10, h - 70), (play_x + play_size[0] + 10, h - 30), (0, 0, 0), -1)
        cv2.putText(display_frame, play_text, (play_x, h - 40), cv2.FONT_HERSHEY_SIMPLEX, font_scale_medium, (255, 255, 0), 2, line_type)

    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("Rock Paper Scissors Game", display_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        logging.info("Game quit by user.")
        break
    elif key == ord(' '):
        if game_over:
            player_score = 0
            computer_score = 0
            round_number = 1
            game_over = False
            round_over = False
            winner_text = ""
            player_move_display = ""
            computer_move_display = ""
            logging.info("New game started.")
            continue

        if round_over:
            if round_number >= MAX_ROUNDS:
                game_over = True
                logging.info(f"Game finished. Final Score: Player {player_score}, Computer {computer_score}")
            else:
                round_number += 1
                winner_text = ""
                player_move_display = ""
                computer_move_display = ""
                logging.info(f"Starting Round {round_number}")
            
            round_over = False
            continue

        if not round_over and not game_over:
            for countdown in range(3, 0, -1):
                ret_cd, frame_cd = cap.read()
                if not ret_cd: continue
                frame_cd = cv2.flip(frame_cd, 1)
                cv2.rectangle(frame_cd, (x1, y1), (x2, y2), (0, 255, 0), 2) # ROI on countdown

                # Center the countdown text
                countdown_text = str(countdown)
                countdown_size = cv2.getTextSize(countdown_text, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)[0]
                countdown_x = (w - countdown_size[0]) // 2
                countdown_y = (h + countdown_size[1]) // 2

                # Add background for better visibility
                cv2.rectangle(frame_cd, 
                             (countdown_x - 20, countdown_y - countdown_size[1] - 20),
                             (countdown_x + countdown_size[0] + 20, countdown_y + 20),
                             (0, 0, 0), -1)
                cv2.putText(frame_cd, countdown_text, (countdown_x, countdown_y), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 4, line_type)
                cv2.imshow("Rock Paper Scissors Game", frame_cd)
                cv2.waitKey(1000)

            ret_pred, frame_pred = cap.read()
            if not ret_pred:
                logging.error("Failed to capture frame for prediction.")
                continue
            frame_pred = cv2.flip(frame_pred, 1)
            roi_predict = frame_pred[y1:y2, x1:x2]

            img_rgb = cv2.cvtColor(roi_predict, cv2.COLOR_BGR2RGB)
            img_resized = cv2.resize(img_rgb, (150, 150))
            img_array = np.expand_dims(img_resized, axis=0)
            img_preprocessed = preprocess_input(img_array.astype(np.float32))

            prediction = model.predict(img_preprocessed, verbose=0)
            predicted_class_idx = np.argmax(prediction)
            confidence = np.max(prediction)

            player_move = CLASS_NAMES[predicted_class_idx] if confidence >= CONFIDENCE_THRESHOLD else "unknown"
            player_move_display = f"{player_move} ({confidence*100:.1f}%)"
            computer_move = get_computer_move()
            computer_move_display = computer_move

            round_winner = get_winner(player_move, computer_move)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            img_filename_base = f'debug_roi_{timestamp}_round{round_number}_conf{confidence:.2f}_{player_move}.jpg'
            img_save_path = os.path.join(LOGS_DIR, img_filename_base)
            try:
                cv2.imwrite(img_save_path, roi_predict)
                logging.info(f'Saved ROI: {img_save_path}')
            except Exception as e:
                logging.error(f"Error saving ROI image '{img_save_path}': {e}")

            logging.info(f'Prediction raw: {prediction}, Player chose: {player_move}, Confidence: {confidence:.4f}')
            logging.info(f'Round {round_number} result: Player={player_move}, Computer={computer_move}, Winner={round_winner}')

            if player_move != "unknown":
                if round_winner == "Player": player_score += 1
                elif round_winner == "Computer": computer_score += 1

            winner_text = f'{round_winner} wins this round!'
            if player_move == "unknown":
                 winner_text = f'Hand not clear ({confidence*100:.1f}%). Try again.'
            elif round_winner == "Draw":
                 winner_text = "It's a Draw this round!"


            round_over = True

cap.release()
cv2.destroyAllWindows()
logging.info("Application closed.")