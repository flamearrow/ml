import matplotlib.pyplot as plt


def draw_first_9(dataset):
    plt.figure(figsize=(10, 10))
    # each take() would emit a batch number of items
    for images, labels in dataset.take(1):
        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(labels.numpy()[i])
            plt.axis("off")
    plt.show()


# draw 30 images and their corresponding labels
# images: n, 244, 244, 3
# labels: n
# if actual_labels is present, color the incorrect labels red
def draw_30_pic_and_labels(images, predicted_labels, actual_labels=None):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(hspace=0.5)
    for i in range(30):
        plt.subplot(6, 5, i + 1)
        plt.imshow(images[i])
        if actual_labels is None:
            color = "black"
        else:
            color = "green" if predicted_labels[i] == actual_labels[i] else "red"
        plt.title(predicted_labels[i], color=color)
        plt.axis("off")
    plt.show()
