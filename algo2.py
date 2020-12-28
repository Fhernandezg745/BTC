import requests

response = requests.get(f'https://api-pub.bitfinex.com/v2/candles/')

data = response.json()

for user in data['results']:
    print(user['name']['first'])


def funcname(greeting, name):
    """[summary]

    Args:
        greeting ([type]): [description]
        name ([type]): [description]

    Returns:
        [type]: [description]
    """

    return f'{greeting} {name}'
