import requests
from bs4 import BeautifulSoup as bs
import numpy as np
import matplotlib.pyplot as plt


input_data = {
    'Всього інфіковано загалом': [],
    'Смертельні випадки загалом': [],
    'Видужали загалом': [],
    'Наразі хворіють': [],
    'Всього інфіковано за день': [],
    'Смертельні випадки за день': [],
    'Видужали за день': [],
}

base_url = 'https://index.minfin.com.ua/ua/reference/coronavirus/ukraine/hmelnickaya'
data = [[], [], [], [], [], [], []]


def create_full_url(url, year, month):
    return url + '/' + str(year) + '-' + str(month).rjust(2, '0') + '/'


def get_values_from_response(response):
    if response.ok:
        soup = bs(response.text, 'lxml')
        tables = [div.table for div in soup.find_all('div',  class_='compact-table')]
        total_table = tables[0]
        for total_info in total_table.find_all('tr'):
            if total_info.find('th'):
                continue
            total_info_arr = total_info.find_all('td', align='right')
            for i in range(4):
                found = total_info_arr[i].contents[0]
                if found.name != 'br':
                    data[i].append(int(found))
                else:
                    data[i].append(0)

        per_day_table = tables[1]
        for per_day_info in per_day_table.find_all('tr'):
            if per_day_info.find('th'):
                continue
            per_day_info = per_day_info.find_all('td', align='right')
            for i in range(3):
                found = per_day_info[i].contents[0]
                if found.name != 'br':
                    data[i+4].append(int(found))
                else:
                    data[i+4].append(0)
    else:
        print("Проблеми зі з'єднанням")


start_year, start_month = 2020, 3
end_year, end_month = 2022, 1
session = requests.Session()
while start_year < end_year or (start_year == end_year and start_month <= end_month):
    full_url = create_full_url(base_url, start_year, start_month)
    request_response = session.get(full_url)
    get_values_from_response(request_response)
    if start_month != 12:
        start_month += 1
    else:
        start_month = 1
        start_year += 1

print(data)

total_infected = np.array(data[0])
total_dead = np.array(data[1])
total_recovered = np.array(data[2])
currently_ill = np.array(data[3])
infected_per_day = np.array(data[4])
dead_per_day = np.array(data[5])
recovered_per_day = np.array(data[6])

plt.plot(total_infected, label='Всього інфіковано загалом', color="blue")
plt.plot(total_dead, label='Смертельні випадки загалом', color="black")
plt.plot(total_recovered, label='Видужали загалом', color="green")
plt.plot(currently_ill, label='Наразі хворіють', color="red")
plt.legend()
plt.show()

plt.plot(infected_per_day, label='Всього інфіковано за день', color="blue")
plt.plot(dead_per_day, label='Смертельні випадки за день', color="black")
plt.plot(recovered_per_day, label='Видужали за день', color="green")
plt.legend()
plt.show()


plt.plot(infected_per_day, label='Всього інфіковано за день', color="blue")
plt.legend()
plt.show()

plt.plot(dead_per_day, label='Смертельні випадки за день', color="black")
plt.legend()
plt.show()

plt.plot(recovered_per_day, label='Видужали за день', color="green")
plt.legend()
plt.show()

