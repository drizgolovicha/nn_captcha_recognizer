import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from neuro_model.index import Net
from utils.image import get, remove_noise


class CustomImages(Dataset):
    data = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def __init__(self):
        self.ord_map = {ord(x): index for index, x in enumerate(self.data, start=1)}
        self.chr_map = dict((v, k) for k, v in self.ord_map.items())

        self.samples = []
        self._init_dataset()

    def __get_img(self):
        """
        get image from site
        :return:
        """
        source = get()

        nparr = np.frombuffer(source, np.uint8)
        image = np.copy(nparr)

        image = cv2.imdecode(image, cv2.IMREAD_UNCHANGED)
        image = image[:, :, [2, 1, 0]]
        image[np.where((image == [34, 34, 34]).all(axis=2) |
                       (image == [33, 33, 33]).all(axis=2) |
                       (image == [32, 32, 32]).all(axis=2))] = [255, 255, 255]

        image[np.where((image != [255, 255, 255]).all(axis=2))] = [0, 0, 0]
        return image

    def _init_dataset(self):
        # get json describes structure
        try:
            image = self.__get_img()
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            cv2.imshow(f"x_", image)
            cv2.waitKey()

            contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                boxes.append((x, y, w, h))

            # filter all trash
            # boxes = list(filter(lambda box: box[2] > 10 and box[3] > 10, boxes))

            # sort found bounds by x coordinate
            boxes.sort(key=lambda i: i[0])

            # do a cut for each number
            for index, (x, y, w, h) in enumerate(boxes):
                im = image[y: y + h, x: x + w].copy()
                im = cv2.resize(im, (64, 64))

                # cv2.imshow(f"x_{x}", im)
                # cv2.waitKey()

                img_tensor = transforms.ToTensor()(im)

                # print(f"img_tensor size {img_tensor}")
                # print(f"letter_tensor size {letter_tensor.size()}")
                self.samples.append(img_tensor)
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
    model_path = "snapshot/net.pt"

    # Model class must be defined somewhere
    model = Net()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    data = CustomImages()
    x = DataLoader(dataset=data, batch_size=6)
    for _, tensor in enumerate(x):
        outputs = model(tensor)
        _, predicted = torch.max(outputs, 1)

        print(predicted)
        predicted_word = [chr(data.chr_map[int(neuron)]) for neuron in predicted]
        print("".join(predicted_word))


if __name__ == "__main__":
    run()
