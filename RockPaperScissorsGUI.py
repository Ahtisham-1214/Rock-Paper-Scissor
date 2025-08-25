# Full GUI Rock-Paper-Scissors with webcam, model prediction, logging, and round management
import cv2
import numpy as np
import random
import logging
from datetime import datetime
import os
import threading
from tkinter import *
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Constants
MODEL_PATH = 'best_model.keras'
LOGS_DIR = "logs"
CLASS_NAMES = ['paper', 'rock', 'scissors']
CONFIDENCE_THRESHOLD = 0.70
MAX_ROUNDS = 3
ROI_SIZE = 250

# Logging setup
if not os.path.exists(LOGS_DIR):
    os.makedirs(LOGS_DIR)
log_file_path = os.path.join(LOGS_DIR, 'game_log.txt')
logging.basicConfig(filename=log_file_path, level=logging.INFO, filemode='a',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Load model
try:
    model = load_model(MODEL_PATH)
    logging.info(f"Model '{MODEL_PATH}' loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError(f"Could not load model: {e}")

# Game state
player_score = 0
computer_score = 0
round_number = 1
game_over = False
frame = None
cap = cv2.VideoCapture(0)

# GUI setup
root = Tk()
root.title("Rock Paper Scissors Game")
root.geometry("1000x800")

video_label = Label(root)
video_label.pack()

status_label = Label(root, text="Welcome!", font=('Helvetica', 14))
status_label.pack(pady=10)

score_label = Label(root, text="", font=('Helvetica', 12))
score_label.pack()

# Helper functions
def get_computer_move():
    return random.choice(CLASS_NAMES)

def get_winner(player, comp):
    if player == "unknown": return "Unknown"
    if player == comp: return "Draw"
    if (player == "rock" and comp == "scissors") or \
       (player == "scissors" and comp == "paper") or \
       (player == "paper" and comp == "rock"):
        return "Player"
    return "Computer"

def update_frame():
    global frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        x1, y1 = w // 2 - ROI_SIZE // 2, h // 2 - ROI_SIZE // 2
        x2, y2 = x1 + ROI_SIZE, y1 + ROI_SIZE

        # Draw ROI box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, 'Place hand in Green Box', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
    root.after(10, update_frame)

def play_round():
    global frame, player_score, computer_score, round_number, game_over
    if frame is None or game_over:
        return

    h, w, _ = frame.shape
    x1, y1 = w // 2 - ROI_SIZE // 2, h // 2 - ROI_SIZE // 2
    x2, y2 = x1 + ROI_SIZE, y1 + ROI_SIZE

    # Countdown before prediction
    for count in range(3, 0, -1):
        ret_cd, frame_cd = cap.read()
        if not ret_cd: continue
        frame_cd = cv2.flip(frame_cd, 1)
        cv2.rectangle(frame_cd, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame_cd, str(count), (w//2 - 20, h//2), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 3)
        img = cv2.cvtColor(frame_cd, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)
        root.update()
        cv2.waitKey(1000)

    ret_pred, frame_pred = cap.read()
    frame_pred = cv2.flip(frame_pred, 1)
    roi = frame_pred[y1:y2, x1:x2]

    # Save ROI
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    roi_path = os.path.join(LOGS_DIR, f'roi_{timestamp}_round{round_number}.jpg')
    cv2.imwrite(roi_path, roi)

    img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (150, 150))
    img = preprocess_input(np.expand_dims(img, axis=0).astype(np.float32))

    prediction = model.predict(img, verbose=0)
    confidence = np.max(prediction)
    player_move = CLASS_NAMES[np.argmax(prediction)] if confidence >= CONFIDENCE_THRESHOLD else "unknown"
    computer_move = get_computer_move()
    winner = get_winner(player_move, computer_move)

    if winner == "Player":
        player_score += 1
    elif winner == "Computer":
        computer_score += 1

    logging.info(f"Round {round_number}: Player={player_move}, CPU={computer_move}, Winner={winner}, Confidence={confidence:.2f}")
    status = f"Player: {player_move} ({confidence*100:.1f}%) | CPU: {computer_move} -> {winner} wins"

    if player_move == "unknown":
        status = f"Hand unclear ({confidence*100:.1f}%). Try again."

    status_label.config(text=status)
    score_label.config(text=f"Round {round_number}/{MAX_ROUNDS} | Player {player_score} - CPU {computer_score}")

    round_number += 1

    if round_number > MAX_ROUNDS:
        game_over = True
        final_msg = "Game Over! "
        if player_score > computer_score:
            final_msg += "You win!"
        elif computer_score > player_score:
            final_msg += "CPU wins!"
        else:
            final_msg += "It's a tie!"
        status_label.config(text=final_msg)
        logging.info(f"Game Over: Final Score Player={player_score}, CPU={computer_score}")

def reset_game():
    global player_score, computer_score, round_number, game_over
    player_score = 0
    computer_score = 0
    round_number = 1
    game_over = False
    status_label.config(text="Game Reset! Press 'Play Round' to begin.")
    score_label.config(text="")

def quit_game():
    cap.release()
    root.destroy()

# GUI Buttons
Button(root, text="Play Round", command=lambda: threading.Thread(target=play_round).start(), font=('Helvetica', 12)).pack(pady=10)
Button(root, text="Reset Game", command=reset_game, font=('Helvetica', 12)).pack(pady=5)
Button(root, text="Quit", command=quit_game, font=('Helvetica', 12)).pack(pady=5)

# Start video loop
update_frame()
root.mainloop()
logging.info("Application closed.")
