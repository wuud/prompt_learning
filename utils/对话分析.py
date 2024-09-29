import requests
from bs4 import BeautifulSoup

class message():
    def __init__(self, owner, time, content):
        self.owner = owner
        self.time = time
        self.content = content


file_path = "C:/Users/Administrator/Desktop/1.txt"

with open(file_path, encoding='utf-8') as f:
    for line in f.readlines():
        print(line)

