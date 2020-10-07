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
