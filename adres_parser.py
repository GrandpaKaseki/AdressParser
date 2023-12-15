import numpy as np
from Levenshtein import distance
from multiprocessing import Manager,Process

# base of symbols to make vectors from words
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
           "-": 34,
           '0': 35,
           '1': 36,
           '2': 37,
           '3': 38,
           '4': 39,
           '5': 40,
           '6': 41,
           '7': 42,
           '8': 43,
           '9': 44
           }


class Parser:

    def __init__(self, street_databse: list | np.ndarray = None,
                 city_database: list | np.ndarray = None,
                 city_key_words: list | set | np.ndarray = {'гор', 'город', 'поселок', 'пос', 'г'},
                 street_key_words: dict = {'ул':'улица','пр':'проспект','ш':'шоссе', 'лин':'линия',
                 'пер':'переулок','бул':'бульвар','мкр':'микрорайон'},
                 house_key_words: list | set | np.ndarray = {'дом', 'д'},
                 separators: set | list | np.ndarray = {';', ':', ',', '.'},
                 k_similar: int = 3) -> None:

        if street_databse is None:
            raise ValueError("street_database parameter is None. Expected ndarray")
        if city_database is None:
            raise ValueError("city_database parameter is None. Expected ndarray")
        self.knn = k_similar
        self.city_database = city_database
        self.street_database = street_databse
        self.city_key_words = city_key_words
        self.street_key_words = street_key_words
        self.house_key_words = house_key_words
        self.all_key_words = self.house_key_words.union(self.street_key_words).union(self.city_key_words).union(self.street_key_words.values())
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

    def _find_probability(self, false_city: str, data_base: list | np.ndarray) -> dict:
        "Looking fot k_similar neighbors for false_city in data_base. Using Levenshtein distance."
        proba_dict = {}
        num_elem = 0
        for city in data_base:
            city = city.lower().replace('�?', 'и')
            dist = distance(false_city, city, weights=(2, 3, 1))
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

    def _find_city_proba(self, city: str) -> dict:
        "Call _find_probabylity then for each finded elemet, calculate exp(-ro), where ro - metrcis in euclidian space."
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
        return proba_dict

    def _find_street_proba(self, street: str, id:int=0, process_dict:dict=None) -> dict | None:
        "Call _find_probabylity then for each finded elemet, calculate exp(-ro), where ro - metrcis in euclidian space."
        input_vector = self._make_vector_from_word(street)
        res_dict = self._find_probability(street, self.street_database)
        proba_dict = dict()
        for i in res_dict.keys():
            for word in res_dict[i]:
                metric_coef = np.exp(-self._find_metric(self._make_vector_from_word(word), input_vector))
                if metric_coef in proba_dict:
                    proba_dict[metric_coef].append(word)
                else:
                    proba_dict[metric_coef] = [word]
        if process_dict is None:
            return proba_dict
        else:
            process_dict[id] = proba_dict
    def _find_tokens(self, line: str) -> list:
        "parse line to tokens. Delete all spec symbols exept '-'"
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
    def _find_index(tokens: list, res_dict: dict) -> None:
        for token in tokens.copy():
            if token.lower().isdigit() and len(token) == 6:
                res_dict['index'] = token
                tokens.remove(token)
                return
        res_dict['index'] = "hasn't identified"

    def _find_city(self, tokens: list, res_dict: dict) -> None:
        "Searching for most similar enter of tokens into city_database"
        key_words = self.city_key_words
        key_word_index = None
        for index, token in enumerate(tokens):
            if token in key_words:
                key_word_index = index
                break
        if type(key_word_index) is int:
            ret_dict_for_one = self._find_city_proba(tokens[key_word_index + 1])
            try:
                ret_dict_for_two = self._find_city_proba(tokens[key_word_index + 1] + " " + tokens[key_word_index + 2])
            except IndexError:
                ret_dict_for_two = {0:0}
            final_dict = {}
            if max(ret_dict_for_one.keys()) > max(ret_dict_for_two.keys()):
                for key in ret_dict_for_one.keys():
                    final_dict[round(key/sum(ret_dict_for_one.keys()),2)] = ret_dict_for_one[key]
                res_dict['city'] = final_dict
                tokens.remove(tokens[key_word_index + 1])
                return
            else:
                for key in ret_dict_for_two.keys():
                    final_dict[round(key / sum(ret_dict_for_two.keys()), 2)] = ret_dict_for_two[key]
                res_dict['city'] = final_dict
                tokens.remove(tokens[key_word_index + 1])
                tokens.remove(tokens[key_word_index + 1])
                return
        else:
            all_possibilities = []
            tokens_for_delete = []
            for i in range(len(tokens)):
                if tokens[i] in self.all_key_words:
                    continue
                all_possibilities.append(self._find_city_proba(tokens[i]))
                tokens_for_delete.append([tokens[i]])
            for i in range(1, len(tokens)):
                if tokens[i] in self.all_key_words or tokens[i - 1] in self.all_key_words:
                    continue
                all_possibilities.append(self._find_city_proba(tokens[i - 1] + " " + tokens[i]))
                tokens_for_delete.append([tokens[i - 1], tokens[i]])
            _max = 0
            index = None
            for i, possibility in enumerate(all_possibilities):
                if max(possibility.keys()) > _max:
                    _max = max(possibility.keys())
                    index = i
            try:
                final_dict = {}
                for key in all_possibilities[index].keys():
                    final_dict[round(key/sum(all_possibilities[index].keys()),2)] = all_possibilities[index][key]
                res_dict['city'] = final_dict
                for delete_token in tokens_for_delete[index]:
                    tokens.remove(delete_token)
                return
            except KeyError:
                raise ValueError("Can't find city... Please check input string")

    def _find_street(self, tokens: list, res_dict: dict) -> None:
        "Searching for most similar enter of tokens into street_database"
        key_words_short = self.street_key_words.keys()
        key_words_full = self.street_key_words.values()
        key_word_index = None
        for index, token in enumerate(tokens):
            if token in key_words_short:
                key_word_index = index
                street_class = self.street_key_words[token]
                break
            elif token in key_words_full:
                key_word_index = index
                street_class = token
                break
        if type(key_word_index) is int:
            proc_arr = []
            return_dict = {}
            for i in range(1,4):
                if key_word_index + i < len(tokens):
                    street = ""
                    for j in range(1,i+1):
                        street += tokens[key_word_index+j] if len(street) == 1 else " "+tokens[key_word_index+j]
                    street += " "+street_class
                    self._find_street_proba(street,i,return_dict)
                if key_word_index - i >= 0:
                    street = ""
                    for j in range(i,0,-1):
                        street += tokens[key_word_index-j] if len(street) == 1 else " "+tokens[key_word_index-j]
                    street += " "+street_class
                    self._find_street_proba(street,-i,return_dict)


            max_ = 0
            id = None
            for i in range(1,4):
                if key_word_index + i < len(tokens):
                    try:
                        if max(return_dict[i].keys()) > max_:
                            max_ = max(return_dict[i].keys())
                            id = i
                    except KeyError:
                        raise UserWarning("произошла ошибочка")
                if key_word_index - i >= 0:
                    try:
                        if max(return_dict[-i].keys()) > max_:
                            max_ = max(return_dict[-i].keys())
                            id = -i
                    except KeyError:
                        raise UserWarning("произошла ошибочка")
            final_dict = {}
            for key in return_dict[id].keys():
                final_dict[round(key/sum(return_dict[id].keys()),2)] = return_dict[id][key]
            res_dict['street'] = final_dict
            delete_tokens = [tokens[key_word_index]]
            if id > 0:
                for j in range(1,id+1):
                    delete_tokens.append(tokens[key_word_index+j])
            else:
                for j in range(-id,0,-1):
                    delete_tokens.append(tokens[key_word_index-j])
            for bad_token in delete_tokens:
                tokens.remove(bad_token)
            return

        else:
            all_possibilities = []
            tokens_for_delete = []
            for i in range(len(tokens)):
                if tokens[i] in self.all_key_words and (tokens[i] not in key_words_short and tokens[i] not in key_words_full):
                    continue
                for street_class in key_words_full:
                    all_possibilities.append(self._find_street_proba(tokens[i] + " " + street_class))
                    tokens_for_delete.append([tokens[i]])
            for i in range(1, len(tokens)):
                if (tokens[i] in self.all_key_words and tokens[i] not in key_words_short and tokens[i] not in key_words_full) or (tokens[i - 1] in self.all_key_words and tokens[i-1] not in key_words_short and tokens[i-1] not in key_words_full):
                    continue
                for street_class in key_words_full:
                    all_possibilities.append(self._find_street_proba(tokens[i - 1] + " " + tokens[i] + " " + street_class))
                    tokens_for_delete.append([tokens[i - 1], tokens[i]])
            for i in range(2, len(tokens)):
                if (tokens[i] in self.all_key_words and tokens[i] not in key_words_short and tokens[
                    i] not in key_words_full) or (
                        tokens[i - 1] in self.all_key_words and tokens[i - 1] not in key_words_short and tokens[
                    i - 1] not in key_words_full) or (
                        tokens[i - 2] in self.all_key_words and tokens[i - 2] not in key_words_short and tokens[
                    i - 2] not in key_words_full):
                    continue
                for street_class in key_words_full:
                    all_possibilities.append(self._find_street_proba(tokens[i-2]+" "+tokens[i - 1] + " " + tokens[i] + " " + street_class))
                    tokens_for_delete.append([tokens[i-2],tokens[i - 1], tokens[i]])
            _max = 0
            index = None
            for i, possibility in enumerate(all_possibilities):
                if max(possibility.keys()) > _max:
                    _max = max(possibility.keys())
                    index = i
            try:
                final_dict = {}
                for key in all_possibilities[index].keys():
                    final_dict[round(key / sum(all_possibilities[index].keys()), 2)] = all_possibilities[index][key]
                res_dict['street'] = final_dict
                for delete_token in tokens_for_delete[index]:
                    tokens.remove(delete_token)
                return
            except KeyError:
                raise ValueError("Can't find city... Please check input string")

    def _find_house(self, tokens: list, res_dict: dict) -> None:
        for token in tokens.copy():
            for char in token:
                if char.isdigit():
                    res_dict['house'] = token
                    tokens.remove(token)
                    return

    def parse_line(self, address_str: str) -> dict:
        """
        Parsing line and returns dict with probability.Dict keys:('index','city','street','house') :arg address_str =
        'улица Дворовая,123456;дом 30,Масква' :returns = {'index': 123456, 'city': {0.82: ['нижний новгород'],
        0.15: ['великий новгород'], 0.03: ['сольвычегодск']}, 'street': {0.12: ['порядковая улица'],
        0.88: ['родниковая улица']}, 'house': 6а}
        """
        tokens = self._find_tokens(address_str)

        res_dict = {'index': None, 'city': None, 'street': None, 'house': None}
        self._find_index(tokens, res_dict)
        self._find_street(tokens, res_dict)
        self._find_house(tokens, res_dict)
        self._find_city(tokens, res_dict)
        for token in tokens:
            if token in self.all_key_words:
                tokens.remove(token)
        return res_dict
