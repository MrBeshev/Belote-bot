import cv2
import time
import random
from ultralytics import YOLO

# ======================
# CONFIG
# ======================
YOLO_MODEL_PATH = "yolocards.pt"
CAMERA_ID = 0

PLAYER_ID = 0
PARTNER_ID = 2
PLAYERS = [0, 1, 2, 3]

# ======================
# BELOTE CONSTANTS
# ======================
RANKS = ["7", "8", "9", "J", "Q", "K", "10", "A"]

TRUMP_ORDER = ["7", "8", "Q", "K", "10", "A", "9", "J"]
NORMAL_ORDER = ["7", "8", "9", "J", "Q", "K", "10", "A"]

TRUMP_POINTS = {"J": 20, "9": 14, "A": 11, "10": 10, "K": 4, "Q": 3}
NORMAL_POINTS = {"A": 11, "10": 10, "K": 4, "Q": 3, "J": 2}

# ======================
# UTILITIES
# ======================
def parse_card(card):
    return card[:-1], card[-1]

def card_value(card, trump, lead):
    rank, suit = parse_card(card)

    if suit == trump:
        return 100 + TRUMP_ORDER.index(rank)
    if suit == lead:
        return NORMAL_ORDER.index(rank)
    return -1

# ======================
# LCD ABSTRACTION
# ======================
class LCD:
    def display(self, line1="", line2=""):
        print(f"[LCD]\n{line1}\n{line2}\n")

# ======================
# VISION
# ======================
class CardDetector:
    def __init__(self):
        self.model = YOLO(YOLO_MODEL_PATH)
        self.cap = cv2.VideoCapture(CAMERA_ID)

    def detect_cards(self, timeout=6):
        detected = set()
        start = time.time()

        while time.time() - start < timeout:
            ret, frame = self.cap.read()
            if not ret:
                continue

            results = self.model(frame, conf=0.5, verbose=False)
            for r in results:
                for c in r.boxes.cls:
                    detected.add(self.model.names[int(c)])

            cv2.imshow("Camera", frame)
            cv2.waitKey(1)

        return list(detected)

# ======================
# GAME STATE
# ======================
class GameState:
    def __init__(self, trump):
        self.trump = trump
        self.played_cards = set()
        self.current_trick = []  # (player, card)

    def record_play(self, player, card):
        self.played_cards.add(card)
        self.current_trick.append((player, card))

    def clear_trick(self):
        self.current_trick.clear()

    def trick_winner(self):
        lead_suit = parse_card(self.current_trick[0][1])[1]
        winner = max(
            self.current_trick,
            key=lambda x: card_value(x[1], self.trump, lead_suit)
        )
        return winner[0]

# ======================
# BELOTE AI
# ======================
class BeloteAI:
    def __init__(self, hand, lcd):
        self.hand = hand
        self.lcd = lcd
        self.trump = None

    def choose_trump(self):
        scores = {s: 0 for s in ["H", "D", "S", "C"]}
        for card in self.hand:
            r, s = parse_card(card)
            scores[s] += TRUMP_POINTS.get(r, 0)

        self.trump = max(scores, key=scores.get)
        self.lcd.display("Trump chosen", self.trump)
        return self.trump

    def play_card(self, game_state):
        trick = game_state.current_trick

        # Determine playable cards
        if trick:
            lead = parse_card(trick[0][1])[1]
            follow = [c for c in self.hand if parse_card(c)[1] == lead]
            playable = follow if follow else self.hand
        else:
            playable = self.hand

        # Partner logic
        if trick:
            winning_player = game_state.trick_winner()
            if winning_player == PARTNER_ID:
                chosen = min(playable, key=lambda c: card_value(c, self.trump, lead))
            else:
                chosen = max(playable, key=lambda c: card_value(c, self.trump, lead))
        else:
            chosen = max(playable, key=lambda c: card_value(c, self.trump, self.trump))

        self.hand.remove(chosen)
        self.lcd.display("Playing card", chosen)
        return chosen

# ======================
# MAIN LOOP
# ======================
def main():
    lcd = LCD()
    detector = CardDetector()

    lcd.display("Show cards", "Scanning...")
    hand = detector.detect_cards()
    lcd.display("Cards detected", " ".join(hand))

    ai = BeloteAI(hand, lcd)
    trump = ai.choose_trump()
    game = GameState(trump)

    current_player = PLAYER_ID

    for trick_no in range(8):
        lcd.display(f"Trick {trick_no+1}", "Playing")

        for _ in range(4):
            if current_player == PLAYER_ID:
                card = ai.play_card(game)
            else:
                card = random.choice(["7H", "8D", "9S", "QC"])  # placeholder

            game.record_play(current_player, card)
            current_player = (current_player + 1) % 4

        winner = game.trick_winner()
        lcd.display("Trick winner", f"Player {winner}")
        game.clear_trick()
        current_player = winner

    lcd.display("Game over", "✔")

if __name__ == "__main__":
    main()
