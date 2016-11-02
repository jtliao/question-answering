import nltk
import pickle
from nltk.tree import Tree

question_types = {
    "Who": 1,
    "When": 2,
    "Where": 3,
    "Who is": 4
}


def get_type_of_question(num_to_question_dict):
    num_to_type_dict = {}
    for num, question in num_to_question_dict.items():
        tokens = nltk.word_tokenize(question)
        question_word = tokens[0]
        if question_word == "Who":
            if tokens[1] == "is" or tokens[1] == "was":
                pos_tagged = nltk.pos_tag(tokens)
                ner_tagged = nltk.ne_chunk(pos_tagged, binary=False)
                if len(ner_tagged) == 4 and ner_tagged[2].label() == "PERSON":
                    num_to_type_dict[num] = question_types["Who is"]
                    # print(question)
                    continue
            num_to_type_dict[num] = question_types["Who"]
        else:
            question_type = question_types[question_word]
            num_to_type_dict[num] = question_type
    return num_to_type_dict


# Gets continuous chunks from nltk trees which is the output of ne_chunk along with their types
# Currently, will split up NE phrases of different types:
#    e.g. New York Yankees is split into ('New York', 'GPE') and ('Yankees', 'ORGANIZATION')
def get_continuous_chunks(chunked):
    continuous_chunk = []
    current_chunk = []
    current_chunk_type = None

    # each i represents subtree or (word, POS) leaf in chunked tree
    for i in chunked:
        if type(i) == Tree:
            if current_chunk_type is None:
                current_chunk.append(" ".join([token for token, pos in i.leaves()]))
                current_chunk_type = i.label()
            else:
                if i.label() == current_chunk_type:
                    current_chunk.append(" ".join([token for token, pos in i.leaves()]))
                else:
                    named_entity = " ".join(current_chunk)
                    if named_entity not in continuous_chunk:
                        continuous_chunk.append((named_entity, current_chunk_type))
                        current_chunk = [(" ".join([token for token, pos in i.leaves()]))]
                        current_chunk_type = i.label()

        elif current_chunk:
            named_entity = " ".join(current_chunk)
            if named_entity not in continuous_chunk:
                continuous_chunk.append((named_entity, current_chunk_type))
                current_chunk = []
                current_chunk_type = None
        else:
            continue

    return continuous_chunk


def get_answers_with_correct_type(num_to_nouns, num_to_type_dict):
    answers = {}
    for question_num, nouns in num_to_nouns.items():
        num_answers = 0
        for doc_num in range(1, 101):
            if num_answers >= 5:
                break

            with open("doc_dev/" + str(question_num) + "/" + str(doc_num) + ".txt") as f:
                print("reading doc " + str(doc_num) + " of question" + str(question_num))
                text = f.read()

                question_type = num_to_type_dict[question_num]

                # NER which can label PERSON, ORAGANIZATION, and GPE (geo political entity)
                # is only useful for filtering out WHO and WHERE questions
                if question_type == 1 or question_type == 3:

                    sentences = nltk.sent_tokenize(text)

                    # Only care about sentences where some noun in question appears
                    for sentence in sentences:
                        for noun in nouns:
                            if noun in sentence:
                                tokens_in_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
                                pos_tagged_tokens_in_sentences = [nltk.pos_tag(sent) for sent in tokens_in_sentences]

                                ner_tagged = nltk.ne_chunk(pos_tagged_tokens_in_sentences, binary=False)
                                ner_chunks = get_continuous_chunks(ner_tagged)

                                #Who questions should get PERSON, or a regular noun... (109: Who is Barbara Jordan)
                                if question_type == 1:
                                    for ner_pair in ner_chunks:
                                        if ner_pair[1] == "PERSON":
                                            answers.append(ner_pair[0])
                                            num_answers += 1

                                # Where questions should get GPE
                                elif question_type == 3:
                                    for ner_pair in ner_chunks:
                                        if ner_pair[1] == "GPE":
                                            answers.append(ner_pair[0])
                                            num_answers += 1


        if num_answers < 5:
            if num_answers == 0:
                answers[question_num] = [(doc_num, "nil")] * 5
            else:
                answers[question_num].append((doc_num, "nil") * (5 - num_answers))
    return answers


def main():
    questions = pickle.load(open("num_to_question_dict.pkl", "rb"))
    print(get_type_of_question(questions))

    string = "New York City is where Derek Jeter and the New York Yankees play."

    # tokens = nltk.word_tokenize(string)
    # tagged = nltk.pos_tag(tokens)
    # named_ent = nltk.ne_chunk(tagged, binary=False)
    # # print(named_ent)
    # print(get_continuous_chunks(named_ent))
    # named_ent.draw()


if __name__ == "__main__":
    main()
