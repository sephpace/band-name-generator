
from torch import Tensor
import yaml


def compile_alphabet(out='data/alphabet.yml'):
    """
    Finds every unique character in every band name and saves it to the given out path.

    Parameters:
    (str) out: The path of the YAML file to save the vocabulary words to.
    """
    with open('data/band_names.yml', 'r') as file:
        data = yaml.full_load(file.read())

    characters = set()
    for key in data:
        for name in data[key]:
            for char in name:
                characters.add(char)

    characters = sorted(list(characters))
    with open(out, 'w') as file:
        file.write(yaml.dump(characters))


def load_alphabet(src='data/alphabet.yml'):
    """
    Loads an alphabet from the given file or compiles a new one if none exists.

    Parameters:
    src (str): The path to the alphabet source.

    Returns:
    (Alphabet): An alphabet object containing words from the source.
    """
    while True:
        try:
            with open(src, 'r') as file:
                data = yaml.full_load(file.read())
                if len(data) == 0:
                    raise FileNotFoundError
                break
        except FileNotFoundError:
            compile_alphabet(src)

    return Alphabet(data)


class Alphabet:
    """
    A dictionary that will return a value if given a key and return the key if
    given a value.

    Attributes:
    characters (list):      A list of alphabet letters.
    __char_to_index (dict): A dictionary that maps letters to indices.
    """
    def __init__(self, characters):
        self.characters = characters
        self.__char_to_index = {}
        for i, letter in enumerate(characters):
            self.__char_to_index[letter] = i

    def __call__(self, value): return self[value]

    def __getitem__(self, key):
        """
        Returns a character if the given key is an index and an index if the given
        character is a key.

        Parameters:
        key (int/str/Tensor): The key

        Returns:
        (str/int): The character/index depending on the type of the key.
        """
        if type(key) == int:
            return self.characters[key]
        elif type(key) == Tensor:
            return self.characters[key.item()]
        else:
            return self.__char_to_index[key]

    def __len__(self): return len(self.characters)

    def __str__(self): return str(self.characters)
