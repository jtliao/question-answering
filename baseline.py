from nltk.tokenize import word_tokenize
from nltk.tag.perceptron import PerceptronTagger

# parts of speech that we are looking for
valid_pos = {"NN", "NNP", "NNS"}
# part of speech tagger that we will use
tagger = PerceptronTagger()


# create map from question number to the question
def parse_question_file(question_file):
    num_to_question = {}
    curr_num = 89
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
def get_nouns_from_questions(questions):
    num_to_nouns_dict = {}
    for num, question in questions.items():
        nouns = []
        tokens = word_tokenize(question)
        tagged = tagger.tag(tokens)
        for word in tagged:
            if word[1] in valid_pos:
                nouns.append(word[0])
        num_to_nouns_dict[num] = nouns
    return num_to_nouns_dict


# get the answers for each question
def get_answers(num_to_nouns):
    # answers = {question num:[(doc num, answer1), (doc num, answer2), ...]}
    answers = {}
    for question_num, nouns in num_to_nouns.items():
        num_answers = 0
        for doc_num in range(1, 101):
            if num_answers >= 5:
                break

            with open("doc_dev/"+str(question_num)+"/"+str(doc_num) + ".txt") as f:
                print("reading doc " + str(doc_num) + "of question" + str(question_num))
                text = f.read()
                # baseline that outputs a length 10 window if one of the nouns in the question is in the window
                tokens = word_tokenize(text)
                tagged = tagger.tag(tokens)
                noun_tagged = [tag[0] for tag in tagged if tag[1] in valid_pos]
                ind = 0
                while ind < len(noun_tagged) - 10 and num_answers < 5:
                    for noun in nouns:
                        if noun in noun_tagged[ind:ind+10]:
                            num_answers += 1
                            answer_string = ""
                            for word in noun_tagged[ind:ind+10]:
                                answer_string += word + " "
                            if question_num in answers:
                                answers[question_num].append((doc_num, answer_string))
                            else:
                                answers[question_num] = [(doc_num, answer_string)]
                            ind += 9
                            break
                    ind += 1
        if num_answers < 5:
            if num_answers == 0:
                answers[question_num] = [(doc_num, "nil")] * 5
            else:
                answers[question_num].append((doc_num, "nil")*(5-num_answers))
    return answers


# write the answers to the text file
def output_answers(answers, answers_file):
    with open(answers_file, "w") as f:
        for question, answer in answers.items():
            for ans in answer:
                f.write(str(question) + " " + str(ans[0]) + " " + ans[1] + "\n")


def main():
    questions = parse_question_file("question.txt")
    num_to_nouns = get_nouns_from_questions(questions)
    answers = get_answers(num_to_nouns)
    output_answers(answers, "answer.txt")


if __name__ == "__main__":
    main()
