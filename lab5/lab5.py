import matplotlib.pyplot as plt
import numpy as np
import requests
from bs4 import BeautifulSoup as bs


def get_formatted_date(year, month):
    return str(year) + '-' + str(month).rjust(2, '0')


def create_full_url(url, date):
    return url + '/' + date + '/'


def segment_by_month(arr):
    d = {}
    for date, value in arr:
        if date not in d:
            d[date] = 0
        d[date] += value
    return list(d.values())


def get_values_from_response(response, date):
    if response.ok:
        soup = bs(response.text, 'lxml')
        tables = [div.table for div in soup.find_all('div',  class_='compact-table')]
        table = tables[-1]
        day_infos = table.find_all('tr')
        for day_info in day_infos:
            if day_info.find('th'):
                continue
            day_info = day_info.find_all('td', align='right')
            for i in range(3):
                found = day_info[i].contents[0]
                value = 0
                if found.name != 'br':
                    value = int(found)
                data[i].append((date, value))
    else:
        print("Проблеми зі з'єднанням")


base_url = 'https://index.minfin.com.ua/ua/reference/coronavirus/ukraine/hmelnickaya'
start_year, start_month = 2020, 3
end_year, end_month = 2022, 1

data = [[], [], []]
session = requests.Session()
while start_year < end_year or (start_year == end_year and start_month <= end_month):
    formatted_date = get_formatted_date(start_year, start_month)
    full_url = create_full_url(base_url, formatted_date)
    request_response = session.get(full_url)
    get_values_from_response(request_response, formatted_date)
    if start_month != 12:
        start_month += 1
    else:
        start_month = 1
        start_year += 1

infected_per_day = np.array([value for date, value in data[0]])
dead_per_day = np.array([value for date, value in data[1]])
recovered_per_day = np.array([value for date, value in data[2]])

infected_per_month = np.array(segment_by_month(data[0]))
dead_per_month = np.array(segment_by_month(data[1]))
recovered_per_month = np.array(segment_by_month(data[2]))

plt.plot(infected_per_day, label="Всього інфіковано за день", color="blue")
plt.legend()
plt.show()

plt.plot(dead_per_day, label="Смертельні випадки за день", color="black")
plt.legend()
plt.show()

plt.plot(recovered_per_day, label="Видужали за день", color="green")
plt.legend()
plt.show()

plt.plot(infected_per_month, label="Всього інфіковано за місяць", color="blue")
plt.legend()
plt.show()

plt.plot(dead_per_month, label="Смертельні випадки за місяць", color="black")
plt.legend()
plt.show()

plt.plot(recovered_per_month, label="Видужали за місяць", color="green")
plt.legend()
plt.show()
