import matplotlib.pyplot as plt


def plot_losses(train_loss, val_loss):
    plt.figure()
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend(); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Loss Curve')
    plt.show()


def plot_metrics(metrics: dict):
    labels = list(metrics.keys())
    values = list(metrics.values())
    plt.figure()
    plt.bar(labels, values)
    plt.title('Model Metrics')
    plt.show()


def plot_complexity(report: dict):
    # visualisasi laporan kompleksitas tergantung tipe
    pass
