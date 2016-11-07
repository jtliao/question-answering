# get rid of the extraneous tags to get the text
def pre():
    for question_num in range(89, 321):
        for doc_num in range(1, 101):
            with open("doc_dev/" + str(question_num) + "/" + str(doc_num), "r") as f:
                with open("doc_dev/" + str(question_num) + "/" + str(doc_num) + ".txt", "w") as f2:
                    print("reading doc " + str(doc_num) + "of question" + str(question_num))
                    file_string = f.read()
                    try:
                        start_text = file_string.index("<TEXT>")
                        end_text = file_string.index("</TEXT>")
                        text = file_string[start_text + 6:end_text]
                        if "[Text]" in text:
                            text = text[text.index("[Text]")+6:]
                        text = text.replace("<P>", "")
                        text = text.replace("</P>", "")
                        f2.write(text)
                    except ValueError:
                        f2.write("")


def main():
    pre()

if __name__ == "__main__":
    main()
