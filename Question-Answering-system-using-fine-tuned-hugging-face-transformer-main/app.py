import gradio as gr
import numpy as np
from transformers import pipeline

title = "Question Answering System"
description = """
Click on examples below to try them

<img src="https://huggingface.co/spaces/course-demos/Rick_and_Morty_QA/resolve/main/rick.png" width=200px>
"""

article = "Check out [my github repository](https://github.com/Neural-Net-Rahul/Question-Answering-system-using-fine-tuned-hugging-face-transformer) and my [fine tuned model](https://huggingface.co/neural-net-rahul/bert-finetuned-squad)"

textbox1 = gr.Textbox(label="Context :", placeholder="The New Yorker noted that by the time Zuckerberg began classes at Harvard in 2002, he had already achieved a reputation as a programming prodigy.", lines=3)
textbox2 = gr.Textbox(label="Question :", placeholder="Who achieved reputation as a programming prodigy?", lines=3)
textbox3 = gr.Textbox(label='Answer :', placeholder="Zuckerberg",lines=5)

model = pipeline('question-answering',model='neural-net-rahul/bert-finetuned-squad')

def ques_ans(context,question):
  return model(question=question, context=context)['answer']


gr.Interface(
    fn=ques_ans,
    inputs=[textbox1,textbox2],
    outputs=textbox3,
    title=title,
    description=description,
    article=article,
    examples=[["Zuckerberg began using computers and writing software in middle school. In high school, he built a program that allowed all the computers between his house and his father's dental office to communicate with each other.","What does the program do?"],["Modi completed his higher secondary education in Vadnagar in 1967; his teachers described him as an average student and a keen, gifted debater with an interest in theatre. He preferred playing larger-than-life characters in theatrical productions, which has influenced his political image","Modi was good in which art?"]]
).launch(share=True)
