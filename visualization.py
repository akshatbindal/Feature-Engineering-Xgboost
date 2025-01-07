import matplotlib.pyplot as plt

def plot_predictions(dates, actual, predicted):
    plt.figure(figsize=(10, 5))
    plt.plot(dates, actual, label='Actual', color='blue')
    plt.plot(dates, predicted, label='Predicted', color='orange')
    plt.legend()
    plt.title("Actual vs Predicted Stock Prices")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid()
    return plt
