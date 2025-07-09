import requests

url = "https://rest.coinapi.io/v1/ohlcv/BINANCEFTS_PERP_BTC_USDT/history?period_id=1DAY&time_start=2020-07-09T02:47:19.000Z&time_end=2020-07-20T02:47:19.000Z"

proxies = {
        'http': "http://user301816:64syt2@45.135.251.31:3179",
        'https': "http://user301816:64syt2@45.150.50.136:4801"
    }

payload = {}
headers = {
  'Accept': 'text/plain',
  'Authorization': '16da381f-0c7c-4f9b-be30-190ccc74e385'
}

response = requests.request("GET", url, headers=headers, data=payload, proxies=proxies)

print(response.text)