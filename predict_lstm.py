"""Load a trained LSTM and predict from seven consecutive daily AQIs."""
import argparse
from lstm_model import load_checkpoint, predict_aqi

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--history', nargs=7, type=float, required=True,
                        help='Seven consecutive observed daily AQIs, oldest first')
    args = parser.parse_args()
    model, saved = load_checkpoint(args.checkpoint)
    print(f'Next-day PM2.5 AQI prediction: {predict_aqi(model, saved, args.history):.4f}')
