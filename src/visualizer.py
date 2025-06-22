import matplotlib.pyplot as plt

class Visualizer:
    """
    Plot loss curves and metrics.
    """
    @staticmethod
    def plot_loss(history: dict):
        plt.figure()
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.legend()
        plt.xlabel('Epoch')
        plt.ylabel('MSE Loss')
        plt.title('Training vs Validation Loss')
        plt.show()
