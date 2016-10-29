import pickle
from nltk.tokenize import word_tokenize
from nltk.tag.perceptron import PerceptronTagger

# look for all the nouns in the question
# look for sentences with the nouns,

valid_pos = {"NN", "NNP", "NNS"}


def get_nouns_from_questions(questions):
    num_to_nouns_dict = {}
    tagger = PerceptronTagger()
    for num, question in questions.items():
        nouns = []
        tokens = word_tokenize(question)
        tagged = tagger.tag(tokens)
        for word in tagged:
            if word[1] in valid_pos:
                nouns.append(word[0])
        num_to_nouns_dict[num] = nouns
    return num_to_nouns_dict


def get_answers(num_to_nouns):
    pass


def main():
    questions = pickle.load(open("num_to_question_dict.pkl", "rb"))
    num_to_nouns = get_nouns_from_questions(questions)
    get_answers(num_to_nouns)


if __name__ == "__main__":
    main()