import nltk
from nltk.tree import Tree
from dateutil import parser
import re
import baseline
from nltk.tag.perceptron import PerceptronTagger
from nltk.stem.lancaster import LancasterStemmer
from queue import PriorityQueue

# parts of speech that we are looking for
valid_pos = {"NN", "NNP", "NNS"}
# part of speech tagger that we will use
tagger = PerceptronTagger()
# find the verb stems
st = LancasterStemmer()

question_types = {
    "Who": 1,
    "Whom": 1,
    "When": 2,
    "Where": 3,
    "Who is": 4
}

months_set = {"January", "February", "March", "April", "May", "June",
              "July", "August", "September", "October", "November", "December"}

bad_verbs = {"is", "was", "did"}


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


def get_answers_with_correct_type_for_question(directory, num_to_type_dict, question_num, nouns, verbs, supers):
    answers = PriorityQueue()
    for doc_num in range(1, 101):
        with open(directory + "/" + str(question_num) + "/" + str(doc_num) + ".txt") as f:
            print("reading doc " + str(doc_num) + " of question" + str(question_num))
            text = f.read()
            question_type = num_to_type_dict[question_num]
            sentences = nltk.sent_tokenize(text)
            doc_answers = PriorityQueue()
            answers_set = set()
            for sentence in sentences:
                curr_answer = []
                num_spots_in_answer = 10
                seen_nouns = set()
                num_verbs = 0
                num_supers = 0
                tokens_in_sentence = nltk.word_tokenize(sentence)
                for token in tokens_in_sentence:
                    if st.stem(token) in verbs:
                        num_verbs += 1
                    elif token in supers:
                        num_supers += 1
                    elif token in nouns:
                        seen_nouns.add(token)
                if len(seen_nouns) > 0:
                    pos_tagged_tokens_in_sentence = nltk.pos_tag(tokens_in_sentence)
                    if question_type == 1 or question_type == 3:
                        ner_tagged = nltk.ne_chunk(pos_tagged_tokens_in_sentence, binary=False)
                        ner_chunks = get_continuous_chunks(ner_tagged)
                        if question_type == 1:
                            for ner_pair in ner_chunks:
                                # Make sure that the person/organization is the same as what we searched for
                                if ner_pair[1] == "PERSON" and ner_pair[0] not in seen_nouns:
                                    tokens = len(nltk.word_tokenize(ner_pair[0]))
                                    if ner_pair[0].lower() in answers_set:
                                        continue
                                    answers_set.add(ner_pair[0].lower())
                                    curr_answer.append(ner_pair[0])
                                    num_spots_in_answer -= tokens
                                    if num_spots_in_answer == 0:
                                        break
                        # Where questions should get GPE
                        elif question_type == 3:
                            for ner_pair in ner_chunks:
                                # Make sure that the location is the same as what we searched for
                                if ner_pair[1] == "GPE" and ner_pair[0] not in seen_nouns:
                                    tokens = len(nltk.word_tokenize(ner_pair[0]))
                                    if ner_pair[0].lower() in answers_set:
                                        continue
                                    answers_set.add(ner_pair[0].lower())
                                    curr_answer.append(ner_pair[0])
                                    num_spots_in_answer -= tokens
                                    if num_spots_in_answer == 0:
                                        break
                    elif question_type == 2:
                        try:
                            for month in months_set:
                                # Detect a month in sentence
                                if month in tokens_in_sentence:
                                    month_ind = tokens_in_sentence.index(month)
                                    # Check for dates in format 2 January
                                    if month_ind > 0 and tokens_in_sentence[month_ind - 1].isdigit():
                                        # Check for dates in format 2 January 2013
                                        if (month_ind < len(tokens_in_sentence) - 1
                                                and tokens_in_sentence[month_ind + 1].isdigit()
                                                and re.match(r"\d{4}$", tokens_in_sentence[month_ind + 1]) is not None):
                                            # There are 3 spots for our answer
                                            if num_spots_in_answer >= 3:
                                                # Appends in month-day-year order
                                                curr_answer.append(tokens_in_sentence[month_ind])
                                                curr_answer.append(tokens_in_sentence[month_ind - 1])
                                                curr_answer.append(tokens_in_sentence[month_ind + 1])
                                                num_spots_in_answer -= 3
                                                if num_spots_in_answer == 0:
                                                    break
                                        else:
                                            if num_spots_in_answer >= 2:
                                                # Appends in month-day order
                                                curr_answer.append(tokens_in_sentence[month_ind])
                                                curr_answer.append(tokens_in_sentence[month_ind - 1])
                                                num_spots_in_answer -= 2
                                                if num_spots_in_answer == 0:
                                                    break
                                    # Check for dates in format January 2
                                    # TODO: allow for letters attached to date number (e.g. January 2nd)
                                    if month_ind < len(tokens_in_sentence) - 1 and tokens_in_sentence[
                                                month_ind + 1].isdigit():
                                        # Check for dates in format January 2 2013
                                        if (month_ind < len(tokens_in_sentence) - 2 and tokens_in_sentence[
                                                month_ind + 1].isdigit()
                                                and tokens_in_sentence[month_ind + 2].isdigit()
                                                and re.match(r"\d{4}$", tokens_in_sentence[month_ind + 1]) is not None):
                                            # There are 3 spots for our answer
                                            if num_spots_in_answer >= 3:
                                                curr_answer.append(tokens_in_sentence[month_ind])
                                                curr_answer.append(tokens_in_sentence[month_ind + 1])
                                                curr_answer.append(tokens_in_sentence[month_ind + 2])
                                                num_spots_in_answer -= 3
                                                if num_spots_in_answer == 0:
                                                    break
                                        else:
                                            if num_spots_in_answer >= 2:
                                                # Appends in month-day order
                                                curr_answer.append(tokens_in_sentence[month_ind])
                                                curr_answer.append(tokens_in_sentence[month_ind + 1])
                                                num_spots_in_answer -= 2
                                                if num_spots_in_answer == 0:
                                                    break
                        # This means that datetime's parser doesn't detect a sentence in here
                        except ValueError:
                            pass
                        for token in tokens_in_sentence:
                            # token is 4 digit (see if it is year)
                            if re.match(r"\d{4}$", token) is not None or re.match(r"\d{4}s$", token) is not None:
                                # Range for valid year based on answer text
                                # Subject to change
                                if (len(token) == 4 and 1500 < int(token) < 2020) or (
                                                len(token) == 5 and 1500 < int(token[:-1]) < 2020):
                                    if token in answers_set:
                                        continue
                                    answers_set.add(token)
                                    curr_answer.append(token)
                                    num_spots_in_answer -= 1
                                    if num_spots_in_answer == 0:
                                        break
                    elif question_type == 4:
                        for (tok, pos) in pos_tagged_tokens_in_sentence:
                            if pos in baseline.valid_pos:
                                answer_noun = tok
                                tokens = len(nltk.word_tokenize(answer_noun))
                                if answer_noun.lower() in answers_set:
                                    continue
                                answers_set.add(answer_noun.lower())
                                curr_answer.append(answer_noun)
                                num_spots_in_answer -= tokens
                                if num_spots_in_answer == 0:
                                    break
                    if curr_answer:
                        doc_answers.put((-len(seen_nouns) * (1 + 2*num_supers) - num_verbs, " ".join(curr_answer)))
        space = 10
        doc_ans = []
        prior = 0
        while doc_answers.qsize() > 0:
            out = doc_answers.get()
            doc_ans.append(out[1])
            prior += out[0]
            space -= 1
            if space == 0:
                answers.put((prior, (doc_num, " ".join(doc_ans))))
                space = 10
                doc_ans = []
                prior = 0
        if doc_ans:
            answers.put((prior, (doc_num, " ".join(doc_ans))))
    if answers.qsize() < 5:
        for ind in range(5 - answers.qsize()):
            answers.put((0, (100, "nil")))
    ans = [answers.get()[1] for _ in range(0, 5)]
    return ans


def get_answers_with_correct_type(directory, num_to_nouns, num_to_verbs, num_to_supers, num_to_type_dict):
    answers = {}
    for question_num, nouns in num_to_nouns.items():
        verbs = num_to_verbs[question_num]
        supers = num_to_supers[question_num]
        answers_for_question = get_answers_with_correct_type_for_question(directory, num_to_type_dict, question_num,
                                                                          nouns, verbs, supers)
        output = []
        for answer in answers_for_question:
            ans = list(answer)
            tokens = nltk.word_tokenize(ans[1])
            if len(tokens) > 10:
                tokens = tokens[:10]
                answer_string = ''.join(word + " " for word in tokens)
                ans[1] = answer_string
                answer = tuple(ans)
            output.append(answer)
        # output.sort(key=lambda t: len(t[1]), reverse=True)
        print(output)
        answers[question_num] = output
    return answers


# create map from question number to the question
def parse_question_file(directory):
    num_to_question = {}
    if directory == "doc_dev":
        curr_num = 89
        question_file = "question.txt"
    else:
        curr_num = 1
        question_file = "question_test.txt"
    next_line_is_descr = False
    with open(question_file, "r") as f:
        for line in f:
            if "<desc>" in line:
                next_line_is_descr = True
            elif next_line_is_descr:
                next_line_is_descr = False
                num_to_question[curr_num] = line
                curr_num += 1
    return num_to_question


# create map from questions to nouns in the question
def get_dicts_from_questions(questions):
    num_to_nouns_dict = {}
    num_to_verbs_dict = {}
    num_to_supers_dict = {}
    for num, question in questions.items():
        nouns = []
        verbs = []
        supers = []
        tokens = nltk.word_tokenize(question)
        tagged = tagger.tag(tokens)
        for word in tagged:
            if word[1] == "JJS" or word[0].lower() == "first":
                if word[0] != "forest":
                    supers.append(word[0])
            elif word[1][0] == "V":
                if word[0] not in bad_verbs:
                    verbs.append(st.stem(word[0]))
            elif word[1] in valid_pos:
                nouns.append(word[0])
        num_to_verbs_dict[num] = verbs
        num_to_supers_dict[num] = supers
        num_to_nouns_dict[num] = nouns
    return num_to_nouns_dict, num_to_verbs_dict, num_to_supers_dict


# write the answers to the text file
def output_answers(answers, answers_file):
    with open(answers_file, "w") as f:
        for question, answer in answers.items():
            for ans in answer:
                f.write(str(question) + " " + str(ans[0]) + " " + ans[1] + "\n")


def main():
    directory = "doc_dev"
    num_to_question = parse_question_file(directory)
    num_to_type_dict = get_type_of_question(num_to_question)
    num_to_nouns_dict, num_to_verbs_dict, num_to_supers_dict = get_dicts_from_questions(num_to_question)

    answers = get_answers_with_correct_type(directory, num_to_nouns_dict, num_to_verbs_dict,
                                            num_to_supers_dict, num_to_type_dict)
    output_answers(answers, "answers_type.txt")


if __name__ == "__main__":
    main()
