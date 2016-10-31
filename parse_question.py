import pickle


def parse_question_file():
    num_to_question = {}
    curr_num = 89
    next_line_is_descr = False
    with open("question.txt", "r") as f:
        for line in f:
            # print(line)
            if "<desc>" in line:
                next_line_is_descr = True
            elif next_line_is_descr:
                next_line_is_descr = False
                num_to_question[curr_num] = line
                curr_num += 1
    # print(num_to_question)
    pickle.dump(num_to_question, open("num_to_question_dict.pkl", "wb"))

def main():
    parse_question_file()

if __name__ == "__main__":
    main()