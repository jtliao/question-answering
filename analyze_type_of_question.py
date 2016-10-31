import nltk
import pickle
import geotext

question_types = {
    "Who": 1,
    "When": 2,
    "Where": 3
}

def get_type_of_question(num_to_question_dict):
    num_to_type_dict = {}
    for num, question in num_to_question_dict.items():
        tokens = nltk.word_tokenize(question)
        question_word = tokens[0]
        question_type = question_types[question_word]
        num_to_type_dict[num] = question_type
    return num_to_type_dict


def main():
    questions = pickle.load(open("num_to_question_dict.pkl", "rb"))
    print(get_type_of_question(questions))
    places = geotext.GeoText("London is a great city")
    places.cities


if __name__ == "__main__":
    main()
