# import PySimpleGUI as sg
# layout = [
#     [sg.Text('Picture Upload 1'), sg.InputText(), sg.FileBrowse(),
#      ],
#     [sg.Text('Picture Upload 2'), sg.InputText(), sg.FileBrowse(),
#      ],
#     [sg.Output(size=(88, 20))],
#     [sg.Submit(), sg.Cancel()]
# ]
# window = sg.Window('Graph Identifier', layout)
# while True:                             # The Event Loop
#     event, values = window.read()
#     # print(event, values) #debug
#     if event in (None, 'Exit', 'Cancel'):
#         break



import gradio as gr


def greet(name):
    return "Hello " + name + "!!"


iface = gr.Interface(fn=greet, inputs="text", outputs="text")
iface.launch()