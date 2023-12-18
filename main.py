import time

from adres_parser import Parser
import pandas as pd


def main():
    city = pd.read_csv("cities_data.csv", sep=" ")
    street = pd.read_csv("streets_new.txt", sep=" ")
    examples = ['улица Дворовая,123456;дом 30a,Масква',
                '6а Ростов-наУлду, ул енина  603132',
                '603132;Нижний Новгород:Голубева,88',
                'Нижний Новгород:Дворовая,д. 30;123456',
                '123456;Ул. Дворовая;30;Москва',
                '123456;НижнийНовгород:ул.Дворовая,30',
                'г. Нижний Новгород, ул. Родниковая, 6а, 123456',
                'гор. НпжнийНовгород, Родниковая 6а, 123456',
                '1-й кемеровский переулок,гор. Ростов-на-Улду,6а']
    work_time = 0
    parser = Parser(street['Streets'], city['Cities'], k_similar=3)
    for adr in examples:
        t1 = time.time()
        result = parser.parse_line(adr)
        work_time += time.time() - t1
        for key in result.keys():
            print(f"{key}: {result[key]}")
        print('\n')
    print(f"время работы {work_time} сек")


def main_input():
    city = pd.read_csv("cities_data.csv", sep=" ")
    street = pd.read_csv("streets_new.txt", sep=" ")
    address = input("Enter your address:\n")
    t1 = time.time()
    parser = Parser(street['Streets'], city['Cities'], k_similar=3)
    result = parser.parse_line(address)
    print(f"время работы {time.time() - t1} сек")
    for key in result.keys():
        print(f"{key}: {result[key]}")
    print('\n')


if __name__ == '__main__':
    main()
    # main_input()
