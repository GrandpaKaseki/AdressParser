from adres_parser import Parser
import pandas as pd


def main():
    city = pd.read_csv("cities_data.csv", sep=" ")
    street = pd.read_csv("streets_new.txt", sep=" ")
    examples = ['Ростов-наУлду, ул Ленина 6а',
                '123456;Нижний Новгород:Дворовая,30',
                'Нижний Новгород:Дворовая,д. 30;123456',
                'улица Дворовая,123456;дом 30,Масква',
                '123456;Ул. Дворовая;30;Москва',
                '123456;НижнийНовгород:ул.Дворовая,30',
                'г. Нижний Новгород, ул. Родниковая, 6а, 123456',
                'гор. НпжнийНовгород, Родниковая 6а, 123456',
                '1-й кемеровский переулок,гор. Ростов-на-Улду,6а']
    parser = Parser(street['Streets'], city['Cities'],k_similar=3)
    for adr in examples:
        result = parser.parse_line(adr)
        for key in result.keys():
            print(f"{key}: {result[key]}")
        print('\n')


if __name__ == '__main__':
    main()
