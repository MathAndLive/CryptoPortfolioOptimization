import requests

url = "https://rest.coinapi.io/v1/symbols/BINANCEFTS"

proxies = {
        'http': "http://user301816:64syt2@45.135.251.31:3179",
        'https': "http://user301816:64syt2@45.150.50.136:4801"
    }

payload = {}
headers = {
  'Accept': 'text/plain',
  'X-CoinAPI-Key': '16da381f-0c7c-4f9b-be30-190ccc74e385'
}

response = requests.request("GET", url, headers=headers, data=payload, proxies=proxies)

print(response.text)