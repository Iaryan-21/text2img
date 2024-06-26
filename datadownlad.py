import requests

urls = [
    "http://images.cocodataset.org/zips/train2017.zip",
    "http://images.cocodataset.org/zips/val2017.zip",
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
]

for url in urls:
    file_name = url.split("/")[-1]
    print(f"Downloading {file_name}...")
    response = requests.get(url)
    with open(file_name, "wb") as f:
        f.write(response.content)
    print(f"{file_name} downloaded.")
