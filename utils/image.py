import random
import matplotlib.pyplot as plt
import numpy as np
import requests
import re

from PIL import Image


def get():
    """
    get new image from destination
    :return:
    """
    def get_headers(cookies):
        return {
            "Cookie": f"{cookies}; _gat=1",
            "Referer": "https://servicesenligne2.ville.montreal.qc.ca/sel/evalweb/index",
            "Sec-Fetch-Mode": "no-cors",
            "Sec-Fetch-Site": "same-origin",
            "Pragma": "no-cache",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"
        }

    source = None
    url_template = "https://servicesenligne2.ville.montreal.qc.ca/sel/evalweb/createimage.png?timestamp={:d}"

    rid = random.randint(10**4, 10**8)
    r = requests.get("https://servicesenligne2.ville.montreal.qc.ca/sel/evalweb/index")

    headers = dict(r.headers)
    x = re.search(r"^[^;]+", headers.get("Set-Cookie"))

    r = requests.get(url_template.format(rid), headers=get_headers(x.group()), stream=True)
    r.raw.decode_content = True
    source = r.raw.data

    print(f"parsed {rid}")
    return source


def remove_noise(img):
    THRESHOLD_VALUE = 254
    with Image.fromarray(img) as im:
        plt.axis("off")
        im.convert("L")
        imgData = np.asarray(im)
        thresholdedData = (imgData > THRESHOLD_VALUE) * 1.0

        plt.imshow(X=thresholdedData)
        plt.show()
