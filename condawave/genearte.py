import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 字体
plt.rcParams['font.family'] = ['Times New Roman',"SimSun"]
plt.rcParams['font.size'] = 20

np.random.seed(40)


# =========================
# 更真实训练数据
# =========================
def generate_training_data(n_epochs=50):

    epochs = np.arange(1, n_epochs + 1)

    # ---- Loss：指数下降 + 抖动 ----
    train_loss = 2.8 * np.exp(-epochs / 12) + 0.4
    train_loss += np.random.normal(0, 0.05, n_epochs)
    train_loss = np.clip(train_loss, 0.2, None)

    # ---- Train Acc：S型增长 ----
    train_acc = 30 + 55 / (1 + np.exp(-0.15 * (epochs - 10)))
    train_acc += np.random.normal(0, 0.6, n_epochs)

    # ---- Test Acc：略低 + 后期轻微下降（模拟过拟合）----
    test_acc = train_acc - np.random.uniform(2, 4, n_epochs)

    # 模拟过拟合（后10epoch）
    test_acc[-10:] -= np.linspace(0, 1.5, 10)

    train_acc = np.clip(train_acc, 30, 86)
    test_acc = np.clip(test_acc, 30, 82)

    return train_loss, train_acc, test_acc


# =========================
# 更真实混淆矩阵
# =========================
def generate_confusion_matrix(n_samples=152000, accuracy=0.79, n_classes=5):

    class_distribution = [0.22, 0.19, 0.21, 0.18, 0.20]
    true_counts = np.array([int(n_samples * p) for p in class_distribution])
    true_counts[-1] += n_samples - true_counts.sum()

    cm = np.zeros((n_classes, n_classes), dtype=int)

    # 某些类更难（更真实）
    class_acc_bias = [0.02, -0.01, 0.0, -0.015, 0.005]

    for i in range(n_classes):

        real_acc = accuracy + class_acc_bias[i]
        correct = int(true_counts[i] * real_acc)
        cm[i, i] = correct

        errors = true_counts[i] - correct

        # 错误分布：邻近类更容易错
        probs = np.random.dirichlet(np.ones(n_classes))
        probs[i] = 0
        probs /= probs.sum()

        for j in range(n_classes):
            if i != j:
                cm[i, j] = int(errors * probs[j])

        # 修正误差
        diff = true_counts[i] - cm[i].sum()
        cm[i, np.random.randint(0, n_classes)] += diff

    return cm


# =========================
# 点线图
# =========================
def plot_training_curve(epochs, train_loss, train_acc, test_acc):

    fig, ax1 = plt.subplots(figsize=(12, 6))

    color1 = '#1f77b4'  # train acc（蓝）
    color2 = '#ff7f0e'  # val acc（橙）
    color3 = '#2ca02c'  # loss（绿）

    # =========================
    # 左轴：Accuracy
    # =========================
    ax1.set_xlabel('Epoch', fontname='Times New Roman')
    ax1.set_ylabel('Accuracy (%)', fontname='Times New Roman', color=color1)

    line1 = ax1.plot(
        epochs, train_acc,
        color=color1,
        linestyle='-',
        marker='o',
        markersize=3,
        linewidth=1.5,
        alpha=0.8,
        label='Train Accuracy'
    )

    line2 = ax1.plot(
        epochs, test_acc,
        color=color2,
        linestyle='-',
        marker='s',
        markersize=3,
        linewidth=1.5,
        alpha=0.8,
        label='Validation Accuracy'
    )

    ax1.tick_params(axis='y', labelcolor=color1, labelsize=20)
    ax1.set_ylim([30, 100])
    ax1.grid(True, alpha=0.3, linestyle='--')

    # =========================
    # 右轴：Loss
    # =========================
    ax2 = ax1.twinx()

    line3 = ax2.plot(
        epochs, train_loss,
        color=color3,
        linestyle='-',
        marker='^',
        markersize=3,
        linewidth=1.5,
        alpha=0.8,
        label='Loss'
    )

    ax2.set_ylabel('Loss', fontname='Times New Roman', color=color3)
    ax2.tick_params(axis='y', labelcolor=color3, labelsize=20)
    ax2.set_ylim([0, 3])  # 根据你数据范围稍微放宽

    # =========================
    # 图例（关键：论文风格）
    # =========================
    lines = line1 + line2 + line3
    labels = [l.get_label() for l in lines]

    ax1.legend(
        lines, labels,
        loc='center right',
        frameon=True,
        fancybox=True,
        shadow=True
    )

    # =========================
    # X轴刻度（和你原图一致）
    # =========================
    ax1.set_xticks(np.arange(0, 51, 5))
    ax1.set_xticklabels(np.arange(0, 51, 5), fontname='Times New Roman')

    # =========================
    # 紧凑布局
    # =========================
    fig.tight_layout()

    plt.show()


# =========================
# 混淆矩阵（增强真实感）
# =========================
def plot_confusion_matrix(cm, class_names):

    fig, ax = plt.subplots(figsize=(9, 7))

    cm_norm = cm / cm.max() * 100

    im = ax.imshow(cm_norm, cmap='Blues')
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))

    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)

    plt.setp(ax.get_xticklabels(), rotation=45)

    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm[i, j]
            pct = val / cm[i].sum() * 100

            color = "white" if cm_norm[i, j] > 50 else "black"

            ax.text(j, i,
                    f'{val:,}\n({pct:.1f}%)',
                    ha='center',
                    va='center',
                    fontsize=18,
                    color=color)

    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.tight_layout()
    plt.show()


# =========================
# 主函数
# =========================
def main():

    epochs = np.arange(1, 51)

    train_loss, train_acc, test_acc = generate_training_data()

    class_names = ['长方体', '球体', '椭圆体', '圆柱体', '正方体']
    cm = generate_confusion_matrix()

    plot_training_curve(epochs, train_loss, train_acc, test_acc)
    plot_confusion_matrix(cm, class_names)


if __name__ == "__main__":
    main()