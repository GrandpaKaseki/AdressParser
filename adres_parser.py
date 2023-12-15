import numpy as np
from Levenshtein import distance

LETTERS = {'й': 0,
           'ц': 1,
           'у': 2,
           'к': 3,
           'е': 4,
           'н': 5,
           'г': 6,
           'ш': 7,
           'щ': 8,
           'з': 9,
           'х': 10,
           'ъ': 11,
           'ф': 12,
           'ы': 13,
           'в': 14,
           'а': 15,
           'п': 16,
           'р': 17,
           'о': 18,
           'л': 19,
           'д': 20,
           'ж': 21,
           'э': 22,
           'я': 23,
           'ч': 24,
           'с': 25,
           'м': 26,
           'и': 27,
           'т': 28,
           'ь': 29,
           'б': 30,
           'ю': 31,
           'ё': 32,
           " ": 33,
           "-": 34
           }


class Parser:

    def __init__(self, street_databse: list | np.ndarray = None,
                 city_database: list | np.ndarray = None, city_key_words: list | set | np.ndarray = None,
                 street_key_words: list | set | np.ndarray = None,
                 house_key_words: list | set | np.ndarray = None, separators: set | list | np.ndarray = None,
                 k_similar: int = 3):
        if separators is None:
            separators = {';', ':', ',', '.'}
        if house_key_words is None:
            house_key_words = {'дом', 'д'}
        if street_key_words is None:
            street_key_words = {'ул', 'улица', 'проспект', 'пр', 'шоссе'}
        if city_key_words is None:
            city_key_words = {'гор', 'город', 'поселок', 'пос', 'г'}
        if street_databse is None:
            raise ValueError("street_database parameter is None. Expected ndarray")
        if city_database is None:
            raise ValueError("city_database parameter is None. Expected ndarray")
        self.knn = k_similar
        self.city_database = street_databse
        self.street_database = street_databse
        self.city_key_words = city_key_words
        self.street_key_words = street_key_words
        self.house_key_words = house_key_words
        self.all_key_words = self.house_key_words.union(self.street_key_words).union(self.city_key_words)
        if type(separators) == set:
            for sep in separators:
                if type(sep) == str:
                    continue
                else:
                    raise TypeError(f"Separator elements must be str type not {type(sep)}")
            self.spec_symbols = separators
        elif type(separators) == list or type(separators) == np.ndarray:
            spec_symbols = set()
            for sep in separators:
                if type(sep) == str:
                    spec_symbols.add(sep)
                else:
                    raise TypeError(f"Separator elements must be str type not {type(sep)}")
        else:
            raise TypeError(f"Expected seperator set or array-like, got {type(separators)} instead")

    def _make_vector_from_word(self, word: str) -> np.ndarray:
        word_vec = np.zeros(len(LETTERS))
        for letter in word:
            word_vec[LETTERS[letter]] += 1
        return word_vec

    def _find_metric(self, vec_a: list | np.ndarray, vec_b: list | np.ndarray) -> float:
        if len(vec_a) != len(vec_b):
            raise ValueError("vectors must be same length")
        ro = 0
        for i in range(len(vec_a)):
            ro += (vec_a[i] - vec_b[i]) ** 2
        return np.sqrt(ro)

    def _find_probability(self, false_city: str, data_base: list|np.ndarray) -> dict:
        proba_dict = {}
        num_elem = 0
        for city in data_base:
            city = city.lower().replace('�?', 'и')
            dist = distance(false_city, city, weights=(2, 2, 1))
            if dist == 0:
                proba_dict[0] = [city]
                return proba_dict
            if dist in proba_dict:
                proba_dict[dist].append(city)
                num_elem += 1
            else:
                proba_dict[dist] = [city]
                num_elem += 1
            if num_elem > self.knn:
                elem = proba_dict.pop(max(proba_dict.keys()))
                num_elem -= len(elem)

        return proba_dict

    def find_city_proba(self, city: str) -> dict:
        input_vector = self._make_vector_from_word(city)
        res_dict = self._find_probability(city, self.city_database)
        proba_dict = dict()
        for i in res_dict.keys():
            for word in res_dict[i]:
                metric_coef = np.exp(-self._find_metric(self._make_vector_from_word(word), input_vector))
                if metric_coef in proba_dict:
                    proba_dict[metric_coef].append(word)
                else:
                    proba_dict[metric_coef] = [word]
        res_dict.clear()
        for metric_coef in proba_dict.keys():
            res_dict[round(metric_coef / sum(proba_dict.keys()), 2)] = proba_dict[metric_coef]
        return res_dict

    def find_street_proba(self, city: str) -> dict:
        res_dict = self._find_probability(city, self.street_database)
        prob_arr = []
        for i in res_dict.keys():
            prob_arr.append(np.exp(-i))
        proba_dict = dict()
        for i, val in enumerate(res_dict.keys()):
            proba_dict[round(prob_arr[i] / sum(prob_arr), 2)] = res_dict[val]
        return proba_dict

    def _find_tokens(self, line: str):
        max_ = 0
        sep = '12krol23'
        for curr_sep in self.spec_symbols:
            joins = line.count(curr_sep)
            if joins > max_:
                line = line.replace(sep, curr_sep)
                sep = curr_sep
        tokens = line.replace(sep, " ").split(sep=" ")
        for i, word in enumerate(tokens):
            tokens[i] = word.strip().lower()
        for _ in range(tokens.count('')):
            tokens.remove('')
        return tokens

    @staticmethod
    def _find_index(tokens: list, res_dict: dict):
        for token in tokens.copy():
            if token.lower().isdigit() and len(token) == 6:
                res_dict['index'] = token
                tokens.remove(token)
                return
        res_dict['index'] = "hasn't identified"

    def _find_city(self, tokens: list, res_dict: dict):
        key_words = {'гор', 'город', 'поселок', 'пос', 'г'}
        key_word_index = None
        for index, token in enumerate(tokens):
            if token in key_words:
                key_word_index = index
                break
        if type(key_word_index) is int:
            ret_dict_for_one = self.find_city_proba(tokens[key_word_index + 1])
            ret_dict_for_two = self.find_city_proba(tokens[key_word_index + 1] + " " + tokens[key_word_index + 2])
            if max(ret_dict_for_one.keys()) > max(ret_dict_for_two.keys()):
                res_dict['city'] = ret_dict_for_one
                tokens.remove(tokens[key_word_index + 1])
                return
            else:
                res_dict['city'] = ret_dict_for_two
                tokens.remove(tokens[key_word_index + 1])
                tokens.remove(tokens[key_word_index + 1])
                return
        else:
            all_possibilities = []
            tokens_for_delete = []
            for i in range(len(tokens)):
                if tokens[i] in self.all_key_words:
                    continue
                all_possibilities.append(self.find_city_proba(tokens[i]))
                tokens_for_delete.append([tokens[i]])
            for i in range(1, len(tokens)):
                if tokens[i] in self.all_key_words or tokens[i - 1] in self.all_key_words:
                    continue
                all_possibilities.append(self.find_city_proba(tokens[i - 1] + " " + tokens[i]))
                tokens_for_delete.append([tokens[i - 1], tokens[i]])
            _max = 0
            index = None
            for i, possibility in enumerate(all_possibilities):
                if max(possibility.keys()) > _max:
                    _max = max(possibility.keys())
                    index = i
            try:
                res_dict['city'] = all_possibilities[index]
                for delete_token in tokens_for_delete[index]:
                    tokens.remove(delete_token)
                return
            except KeyError:
                raise ValueError("Can't find city... Please check input string")

    def _find_house(self, tokens: list, res_dict: dict):
        for token in tokens.copy():
            for char in token:
                if char.isdigit():
                    res_dict['house'] = token
                    tokens.remove(token)
                    return

    def parse_line(self, address_str: str):
        tokens = self._find_tokens(address_str)

        res_dict = {'index': None, 'city': None, 'street': None, 'house': None}
        self._find_index(tokens, res_dict)
        self._find_house(tokens, res_dict)
        self._find_city(tokens, res_dict)
        for token in tokens:
            if token in self.all_key_words:
                tokens.remove(token)
        res_dict['street'] = tokens
        return res_dict
