import asyncio
import shutil
import time
import aiohttp
import random
import requests


async def create_session(headers):
    conn = aiohttp.TCPConnector(limit_per_host=5)
    return aiohttp.ClientSession(headers=headers, connector=conn)


async def get(session, http_url: str):
    """
    get data async way
    :param session:
    :param http_url: str
    :return: List[Dict]
    """
    print(f"Get data from {http_url}")

    try:
        async with session.get(http_url, raise_for_status=True) as x:
            return await x.read()
    except Exception as e:
        raise


def get_headers():
    return {
        "Cookie": "JSESSIONID=XBt7iB-a-uKDhLyYWVVOw7TyBSEEfBW0ZIVm4Uf-x0w-evrCWh0x!-209918963; _gat=1",
        "Referer": "https://servicesenligne2.ville.montreal.qc.ca/sel/evalweb/index",
        "Sec-Fetch-Mode": "no-cors",
        "Sec-Fetch-Site": "same-origin",
        "Pragma": "no-cache",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_2) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/79.0.3945.88 Safari/537.36"
    }


def __run():
    url_template = "https://servicesenligne2.ville.montreal.qc.ca/sel/evalweb/createimage.png?timestamp={:d}"
    s = asyncio.get_event_loop().run_until_complete(create_session(headers=get_headers()))

    coroutines = [get(session=s, http_url=url_template.format(random.randint(10**4, 10**8))) for x in range(0, 1)]
    # coroutines = [get(session=s, http_url=url_template.format(1578325516189)) for x in range(0, 1)]

    # get initial data from server to prepare list
    loop = asyncio.get_event_loop()

    results = loop.run_until_complete(asyncio.gather(*coroutines))
    loop.run_until_complete(s.close())

    for result in results:
        with open(f"data/{random.randint(10**4, 10**8)}.png", "wb") as f:
            f.write(result)


def run():
    url_template = "https://servicesenligne2.ville.montreal.qc.ca/sel/evalweb/createimage.png?timestamp={:d}"
    for _ in range(0, 100):
        rid = random.randint(10**4, 10**8)
        requests.get("https://servicesenligne2.ville.montreal.qc.ca/sel/evalweb/index", headers=get_headers())
        r = requests.get(url_template.format(rid), headers=get_headers(), stream=True)
        with open(f"data/{rid}.png", "wb") as f:
            r.raw.decode_content = True
            shutil.copyfileobj(r.raw, f)

        print(f"parsed {rid}")
        time.sleep(1)


if __name__ == "__main__":
    run()
