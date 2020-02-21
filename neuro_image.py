import json
import os
import cv2
import \
    torch
import random
from neuro_model.index import Net
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets
import torch.optim as optim
import torch.nn as nn


class CustomImages(Dataset):
    data = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def __init__(self, data_root, data_type="train"):
        self.ord_map = {ord(x): index for index, x in enumerate(self.data, start=1)}
        # print(f"ord map is {self.ord_map}")

        self.data_root = data_root
        self.data_type = data_type
        self.samples = []
        self._init_dataset()

    def _init_dataset(self):
        # get json describes structure
        try:
            with open(os.path.join(self.data_root, "desc.json")) as f:
                config = json.loads(f.read())

            nodes = config[self.data_type]
            random.shuffle(nodes)

            nodes_amount = len(nodes)
            limit = int(nodes_amount * 0.8) if self.data_type == "train" else int(nodes_amount * 0.2)

            for node in nodes[:limit]:
                image = cv2.imread(f"data_prepared_bak/{node.get('name')}", cv2.IMREAD_GRAYSCALE)
                # image = cv2.imread(f"data_prepared_bak/15033728.png", cv2.IMREAD_GRAYSCALE)
                image = cv2.bitwise_not(image)

                contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                """
                cv2.imshow(f"x_", image)
                cv2.waitKey()
                exit(1)
                """
                boxes = []

                for c in contours:
                    x, y, w, h = cv2.boundingRect(c)
                    boxes.append((x, y, w, h))
                    """
                    # cv2.rectangle(image, (x, y), (x + w, y + h), 255, 0)
                    # print(transforms.ToTensor()(image[y: y + h, x: x + w]).shape)
                    im = image[y: y + h, x: x + w].copy()
                    print(x)
                    im = cv2.resize(im, (64, 64))
                    cv2.imshow(f"x_{x}", im)
                    self.samples.append((transforms.ToTensor()(im), torch.tensor([ord(c) for c in node["text"]])))
                    """

                # filter all trash
                boxes = list(filter(lambda box: box[2] > 10 and box[3] > 10, boxes))

                # sort found bounds by x coordinate
                boxes.sort(key=lambda i: i[0])

                # do a cut for each number
                for index, (x, y, w, h) in enumerate(boxes):
                    im = image[y: y + h, x: x + w].copy()
                    im = cv2.resize(im, (64, 64))
                    """
                    print(node["text"])
                    cv2.imshow(f"x_{x}", im)
                    cv2.waitKey()
                    """

                    try:
                        ord_letter = ord(node["text"][index])
                    except BaseException as e:
                        pass

                    img_tensor = transforms.ToTensor()(im)
                    # print(self.ord_map.get(ord_letter), node["text"][index])
                    letter_tensor = torch.tensor(self.ord_map.get(ord_letter))

                    # print(f"img_tensor size {img_tensor}")
                    # print(f"letter_tensor size {letter_tensor.size()}")
                    self.samples.append((img_tensor, letter_tensor))

                # image = cv2.drawContours(image, contours[0:3], -1, (0, 0, 0), 3)

                # im = Image.open(os.path.join(self.data_root, node["name"]))
                # convert to 64x64, 8 bit
                # im = im.convert(mode="1", colors=2).resize(size=(64, 64))
                # im.save(os.path.join(self.data_root, node["name"]), format="PNG", quality=1, optimize=True, progressive=True)

        except BaseException:
            raise

    def __len__(self) -> int:
        """
        get size of the data-set
        :return: int
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """
        init getitem method
        :param idx: int
        :return: tuple
        """
        return self.samples[idx]


def run():
    # convert image to tensor
    data_dir = "data_prepared"
    train_images = CustomImages(data_root=data_dir, data_type="train")
    test_images = CustomImages(data_root=data_dir, data_type="test")

    train_loader = DataLoader(dataset=train_images, batch_size=6, num_workers=2)
    test_loader = DataLoader(dataset=test_images, batch_size=6, num_workers=2)

    learning_rate = 0.001
    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    # Train the model
    total_step = len(train_loader)
    loss_list = []
    acc_list = []
    num_epochs = 5

    for epoch in range(num_epochs):  # loop over the data-set multiple times

        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader, 0):
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track the accuracy
            total = labels.size(0)

            _, predicted = torch.max(outputs.data, 1)

            correct = (predicted == labels).sum().item()
            acc_list.append(correct / total)

            if (i + 1) % 5 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                              (correct / total) * 100))

    print('Finished Training')

    net.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in test_loader:
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f"Test Accuracy of the model on the {len(test_loader)} test images: {(correct / total) * 100} %")

    # torch.save(net.state_dict(), "snapshot/net.pt")


if __name__ == "__main__":
    run()
