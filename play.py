"""Play Connect Four against a trained checkpoint from the terminal."""
import argparse
from pathlib import Path

import torch

from agents import MCTSAgent
from game import Connect4
from network import NetworkEvaluator, PolicyValueNet

CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="best.pt")
    parser.add_argument("--simulations", type=int, default=200)
    parser.add_argument("--human-first", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    net = PolicyValueNet().to(device)
    net.load_state_dict(torch.load(CHECKPOINT_DIR / args.checkpoint, map_location=device))
    net.eval()
    ai = MCTSAgent(NetworkEvaluator(net, device), n_simulations=args.simulations)

    game = Connect4()
    human_player = 1 if args.human_first else -1
    print(f"You are {'X' if human_player == 1 else 'O'}. Columns are 0-6.\n")

    while not game.is_terminal():
        print(game)
        print()
        if game.current_player == human_player:
            legal = game.legal_moves()
            move = None
            while move not in legal:
                try:
                    move = int(input(f"Your move {legal}: "))
                except ValueError:
                    continue
        else:
            print("AI is thinking...")
            move = ai.choose_move(game)
            print(f"AI plays column {move}")
        game = game.apply_move(move)
        print()

    print(game)
    winner = game.check_winner()
    if winner == 0:
        print("\nDraw!")
    elif winner == human_player:
        print("\nYou win!")
    else:
        print("\nAI wins!")


if __name__ == "__main__":
    main()
