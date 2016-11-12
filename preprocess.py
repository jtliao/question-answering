import os
import re

# get rid of the extraneous tags to get the text
def pre(target_dir):
    for question_num in os.listdir(target_dir):
        for doc_num in os.listdir(os.path.join(target_dir, question_num)):
            if "txt" in doc_num:
                continue
            with open(target_dir + "/" + str(question_num) + "/" + str(doc_num), "r") as f:
                with open(target_dir + "/" + str(question_num) + "/" + str(doc_num) + ".txt", "w") as f2:
                    print("reading doc " + str(doc_num) + "of question" + str(question_num))
                    file_string = f.read()
                    try:
                        start_text_lst = [m.start() for m in re.finditer("<TEXT>", file_string)]
                        end_text_lst = [m.start() for m in re.finditer("</TEXT>", file_string)]

                        if len(start_text_lst) != len(end_text_lst):
                            print("NOT EQUAL")
                            return

                        for i in range(len(start_text_lst)):
                            start_text = start_text_lst[i]
                            end_text = end_text_lst[i]
                            text = file_string[start_text + 6:end_text]
                            if "[Text]" in text:
                                text = text[text.index("[Text]")+6:]
                            text = text.replace("<P>", "")
                            text = text.replace("</P>", "")
                            f2.write(text)
                    except ValueError:
                        f2.write("")


def main():
    pre("doc_dev")

if __name__ == "__main__":
    main()
